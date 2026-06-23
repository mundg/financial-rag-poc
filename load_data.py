import pandas as pd
import ast
import psycopg2
from psycopg2.extras import execute_values,execute_batch
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.vertex import VertexTextEmbedding
import google.auth
from dotenv import load_dotenv 
import urllib
import os

# Load data from CSV source 
df = pd.read_csv('data/gretel_financial_risk.csv')

# Google Credentials
credentials, project_id = google.auth.default()
    

class ingest_PGVector:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")
        db_host = os.environ.get("DB_HOST")
        db_port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")

        if db_host and db_host.startswith("/cloudsql/"):
            print("Connecting via Cloud SQL internal Unix sockets...")
            self.conn = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host 
            )
        else:
            self.DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            self.conn = psycopg2.connect(self.DATABASE_URL)
            print("Connecting via local TCP network loopback...")
        print("Connecting securely to the PostgreSQL cluster...")
        
        self.embed_model = VertexTextEmbedding(
            model_name="text-embedding-004",
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_REGION", "us-central1"),
            credentials=credentials
        )

        # self.splitter = SemanticSplitterNodeParser(
        #     buffer_size=1, 
        #     breakpoint_percentile_threshold=95, 
        #     embed_model=self.embed_model
        # )

        self.splitter = SentenceSplitter(
            chunk_size=512,      
            chunk_overlap=32     
        )

    def __del__(self):
        """Safely close database socket when the workflow session terminates"""
        try:
            self.conn.close()
        except Exception:
            pass

    
    # Chunking Strategy (Semantic Chunking + previous/next node linking)
    def _chunking(self, df: pd.DataFrame):
        all_nodes_list = []

        print("Executing chunking...") 
        for index, row in df.iterrows():  

            try:
                summary_dict = ast.literal_eval(row["target_summary"])
                analysis_summary = summary_dict.get("analysis", "")
            except Exception:
                analysis_summary = row["target_summary"]

            combined_text = f"SUMMARY ANALYSIS: {analysis_summary}\n\nRAW REPORTING: {row['raw_text']}"

            doc = Document(
                text=combined_text,
                metadata={
                    "ticker": row["ticker"],
                    "fiscal_year": int(row["fiscal_year"]),
                    "risk_category": row["risk_category"],
                    "severity": row["severity"]
                }
            )
            
            # Slice the single row document into semantic nodes in absolute isolation
            row_nodes = self.splitter.get_nodes_from_documents([doc])
            
            # Accumulate the nodes into our master list
            for i, node in enumerate(row_nodes):
                db_node_id = node.node_id
                prev_pointer = row_nodes[i-1].node_id if i > 0 else None
                vector_embedding = self.embed_model._get_text_embedding(node.get_content())
                
                all_nodes_list.append({
                    "node_id": db_node_id,
                    "chunk_text": node.get_content(),
                    "ticker": node.metadata["ticker"],
                    "fiscal_year": node.metadata["fiscal_year"],
                    "risk_category": node.metadata["risk_category"],
                    "severity": node.metadata["severity"],
                    "embedding": vector_embedding,
                    "prev_node_id": prev_pointer,
                    "next_node_id": None 
                })

        # ==========================================
        # Next-node linking
        # ==========================================

        for idx in range(len(all_nodes_list) - 1):
            # CRITICAL SECURITY GUARDRAIL: Only link if tickers match
            if all_nodes_list[idx]["ticker"] == all_nodes_list[idx+1]["ticker"]:
                all_nodes_list[idx]["next_node_id"] = all_nodes_list[idx+1]["node_id"]

        return all_nodes_list
    

    def _write_data(self, all_nodes_list: list):
        if not all_nodes_list:
            print("No entries to write.")
            return

        try:
            self.conn.rollback() 
            with self.conn.cursor() as cursor:
                
                # ─── PHASE 1: RUN SCHEMA INITIALIZATION ONCE ───
                print("Rebuilding database table target schemas...")
                setup_query = """
                    DROP TABLE IF EXISTS financial_analysis_chunks;
                    CREATE TABLE IF NOT EXISTS financial_analysis_chunks (
                        node_id TEXT PRIMARY KEY,
                        chunk_text TEXT,
                        ticker TEXT,
                        fiscal_year INTEGER,
                        risk_category TEXT,
                        severity TEXT,
                        embedding VECTOR(768),
                        prev_node_id TEXT,
                        next_node_id TEXT
                    );
                """
                cursor.execute(setup_query)

                # ─── PHASE 2: COMPILE ACCURATE RECORD TUPLES ───
                base_records = [
                    (
                        n["node_id"], 
                        n["chunk_text"], 
                        n["ticker"],          
                        n["fiscal_year"],      
                        n["risk_category"],    
                        n["severity"],         
                        n["embedding"], 
                        n["prev_node_id"], 
                        n["next_node_id"]
                    )
                    for n in all_nodes_list
                ]

                # ─── PHASE 3: DISTRIBUTED CHUNKED BATCH INSERTS ───
                BATCH_SIZE = 200
                total_records = len(base_records)
                print(f"Streaming data into Cloud SQL in chunks of {BATCH_SIZE}...")
                
                pure_insert_query = """
                    INSERT INTO financial_analysis_chunks (
                        node_id, chunk_text, ticker, fiscal_year, risk_category, severity, embedding, prev_node_id, next_node_id
                    ) VALUES %s;
                """
                
                # Loop through the list in steps of 200
                for start_idx in range(0, total_records, BATCH_SIZE):
                    end_idx = start_idx + BATCH_SIZE
                    batch_slice = base_records[start_idx:end_idx]
                    
                    # Stream the specific batch payload over the proxy tunnel
                    execute_values(cursor, pure_insert_query, batch_slice)
                    print(f"Successfully flushed batch rows {start_idx} to {min(end_idx, total_records)}.")

            # Commit the entire sequence successfully at the absolute end
            self.conn.commit()
            print("All batches committed! Database is fully up to date.")
            
        except Exception as e:
            print(f"Error during chunked database write transaction: {e}")
            self.conn.rollback()
        
        

if __name__ == "__main__":
    pipeline = ingest_PGVector()
    nodes_list = pipeline._chunking(df)
    write_db = pipeline._write_data(nodes_list)