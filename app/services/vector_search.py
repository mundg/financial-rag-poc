from pathlib import Path
import os
import json
import re
import psycopg2
import google.auth
from google import genai
from google.genai import types
from dotenv import load_dotenv
from llama_index.embeddings.vertex import VertexTextEmbedding
import urllib



credentials, project_id = google.auth.default()
script_dir = Path(__file__).resolve().parent
with open(script_dir / "config_metadata/metadata_config.json", "r") as f:
    config = json.load(f)

TICKER_MAPPINGS = config["ticker_mappings"]
VALID_YEARS = config["valid_years"]

class FinancialRAGWorkflow:
    def __init__(self):
        load_dotenv()
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

        

        # google genAI LLM client
        self.client = genai.Client(
            vertexai=True, credentials=credentials
        )

        self.embed_model = VertexTextEmbedding(
            model_name="text-embedding-004",
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_REGION", "us-central1"),
            credentials=credentials
        )
        
        self.chat_history = [] 
        self.max_memory_turns = 4

    def __del__(self):
        """Safely close database socket when the workflow session terminates"""
        try:
            self.conn.close()
        except Exception:
            pass

    def _determine_routing_intent(self, user_query: str) -> str:
        # 1. Direct conversational filler/greetings
        conversational_signals = [
            r"\b(hey|hello|hi|thanks|thank you|cool|ok|okay)\b",
            r"\b(agree|makes sense|that's right|exactly|perfect|awesome)\b",
            r"\b(yes|no|yup|nah|test)\b$" 
        ]
        
        # 2. 🚀 NEW: Ambiguous follow-up commands that rely entirely on recent history
        follow_up_signals = [
            r"\b(elaborate|explain|expand|tell me more|what do you mean)\b",
            r"\b(clarify|go deeper|summarize that)\b"
        ]
        
        # Check direct conversational signals
        for pattern in conversational_signals:
            if re.search(pattern, user_query, re.IGNORECASE) and len(user_query.split()) < 5:
                return "CHAT"
                
        # Check follow-up command signals (even if the sentence is long!)
        for pattern in follow_up_signals:
            if re.search(pattern, user_query, re.IGNORECASE):
                if len(self.chat_history) > 0:
                    print("Fast-Pass: Follow-up intent detected. Routing to chat memory history.")
                    return "CHAT"
                
        # Fast-pass 3: If it explicitly matches config file entities, force RAG lookup
        fallback = self.config_driven_fallback(user_query)
        if fallback["tickers"] or fallback["fiscal_years"]:
            return "RAG"
            
        return "RAG"

    def config_driven_fallback(self, text: str):
        extracted_tickers = []
        extracted_years = []
        
        for key, canonical_ticker in TICKER_MAPPINGS.items():
            if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                extracted_tickers.append(canonical_ticker)
                
        for year in VALID_YEARS:
            if re.search(r'\b' + str(year) + r'\b', text):
                extracted_years.append(str(year))
                
        return {
            "tickers": extracted_tickers,
            "fiscal_years": extracted_years
        }

    def _get_metadata_filters(self, user_query):
        fallback = self.config_driven_fallback(user_query)
        raw_tickers = fallback.get("tickers", [])
        raw_years = fallback.get("fiscal_years", [])
        
        if raw_tickers or raw_years:
            filter_tickers = raw_tickers
            filter_years = [int(year) for year in raw_years if str(year).isdigit()]
            print(f"Fast-pass regex triggered! Bypassed local LLM extraction.")
            return filter_tickers, filter_years
            
        routing_prompt = f"""[Task] Extract mentioned stock tickers and 4-digit fiscal years from the User Query into the exact JSON format specified below.
        [Formatting Rule]
        Output ONLY a valid JSON object matching this empty data type schema. Do not copy placeholder names.
        {{
          "tickers": [],
          "fiscal_years": []
        }}
        User Query: {user_query}
        [JSON Output]"""

        filter_tickers = []
        filter_years = []
        
        try:
            print("Routing query to LLM for structured metadata extraction...")
            llm_response = self.client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=routing_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    )
                ).text.strip()
            print("gemini_response\n",llm_response)
            llm_response = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL)
            json_match = re.search(r'\{.*\}', llm_response, flags=re.DOTALL)

            if json_match:
                llm_response = json_match.group(0)
            
            parsed_json = json.loads(llm_response)
            filter_tickers = parsed_json.get("tickers", [])
            extracted_years = parsed_json.get("fiscal_years", [])
            filter_years = [int(year) for year in extracted_years if str(year).isdigit()]
            print("metadata pickup:\n",filter_years, filter_tickers)
        except Exception as e:  
            print(f"Error during LLM metadata extraction: {e}")
        
        return filter_tickers, filter_years
    
    def _query(self, user_query: str, k_rows: int = 3):
        filter_tickers, filter_years = self._get_metadata_filters(user_query)
        query_vector = self.embed_model.get_text_embedding(user_query)
        
        # ⚡ OPTIMIZATION: Use Left Joins to pull neighbor text data concurrently on pass 1!
        vector_search_query = f"""
            SELECT 
                main.node_id, 
                main.chunk_text AS target_text, 
                main.ticker, 
                main.fiscal_year,
                prev.chunk_text AS prev_text, 
                next.chunk_text AS next_text
            FROM financial_analysis_chunks main
            LEFT JOIN financial_analysis_chunks prev ON main.prev_node_id = prev.node_id
            LEFT JOIN financial_analysis_chunks next ON main.next_node_id = next.node_id
            WHERE 
                (coalesce(cardinality(%s::TEXT[]), 0) = 0 OR main.ticker = ANY(%s))
                AND (coalesce(cardinality(%s::INTEGER[]), 0) = 0 OR main.fiscal_year = ANY(%s))
            ORDER BY main.embedding <=> %s::vector
            LIMIT {k_rows};
        """

        with self.conn.cursor() as cursor:
            cursor.execute(vector_search_query, (
                filter_tickers, filter_tickers, 
                filter_years, filter_years, 
                query_vector
            ))
            matched_rows = cursor.fetchall()
        return matched_rows
    
    def _generate_llm_context(self, matched_rows):
        final_llm_context = ""
        
        for row in matched_rows:
            # Unpack the pre-fetched neighbor strings directly from your SQL statement
            m_node_id, m_text, m_ticker, m_year, previous_chunk_text, next_chunk_text = row
            
            final_llm_context += f"--- Document Section Partition: {m_ticker} ({m_year}) ---\n"
            
            if previous_chunk_text: 
                final_llm_context += f"[Prior Chronological Context]:\n{previous_chunk_text}\n\n"
                
            final_llm_context += f"[Target Direct Context]:\n{m_text}\n\n"
            
            if next_chunk_text: 
                final_llm_context += f"[Subsequent Chronological Context]:\n{next_chunk_text}\n"
                
            final_llm_context += "==================================================\n\n"

        return final_llm_context
    
    def answer_query(self, user_query: str):
        intent = self._determine_routing_intent(user_query)
        
        # --- PATHWAY A: LIGHTWEIGHT CONVERSATIONAL CHAT ---
        if intent == "CHAT":
            print("Fast-Pass: Bypassing pgvector search.")
            
            history_context = ""
            for turn in self.chat_history:
                history_context += f"{turn['role'].capitalize()}: {turn['content']}\n"
                
            chat_prompt = f"""You are a financial analyst tracking a conversation. 
            Review the history, then fulfill the user's latest request by elaborating on or clarifying the previous points.

            [Conversation History]
            {history_context}
            User: {user_query}
            Assistant:"""
            

            response = self.client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=chat_prompt
                ).text.strip()
            
            # Keep history moving forward
            self.chat_history.append({"role": "user", "content": user_query})
            self.chat_history.append({"role": "assistant", "content": response})
            print("Chat response.")
            return response
        
        # --- PATHWAY B: STANDARDIZED FACTUAL RAG ROUTINE ---
        matched_rows = self._query(user_query)
        if not matched_rows:
            return "I searched the database but didn't find any relevant insights for that query. Try asking something like: 'What was Apple's revenue in 2026?'"

        final_llm_context = self._generate_llm_context(matched_rows)

        final_prompt_template = f"""You are a professional financial analyst. Synthesize a concise answer using the following chronological context blocks.
        {final_llm_context}
        User Question: {user_query}
        Answer:"""
        
        final_llm_response = self.client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=final_prompt_template
            ).text.strip()
        print("Gemini llm financial response.")
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": final_llm_response})

        if len(self.chat_history) > self.max_memory_turns * 2:
            self.chat_history = self.chat_history[-self.max_memory_turns * 2:]
        

        return final_llm_response


if __name__ == "__main__":
    workflow = FinancialRAGWorkflow()
    print("System active. Type your financial queries below (Type 'exit' to quit):")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            print("Thinking...")
            output = workflow.answer_query(user_input)
            print(f"Assistant: {output}")
        except KeyboardInterrupt:
            break