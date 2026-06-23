import gradio as gr
import time
from services.vector_search import FinancialRAGWorkflow 

workflow = FinancialRAGWorkflow()
PROMPT_COUNT = 0
PROMPT_LIMIT = 25

def execute_gradio_pipeline(user_message, chat_history):
    global PROMPT_COUNT
    
    # 🚨 1. THE GATEKEEPER CHECK
    if PROMPT_COUNT >= PROMPT_LIMIT:
        warning_message = "..."
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": warning_message})
        
        yield chat_history  # Sends the warning message to the UI screen
        return
    
    PROMPT_COUNT += 1
    print(f"📈 Cloud Prompt Initiated. Allocation: {PROMPT_COUNT}/{PROMPT_LIMIT}")

    pipeline_result = workflow.answer_query(user_message)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    
    accumulator = ""
    for character in pipeline_result:
        accumulator += character
        chat_history[-1]["content"] = accumulator
        time.sleep(0.005)  # Quick typing effect interval
        yield chat_history



with gr.Blocks(title="RAG Analytics Portal") as demo:
    gr.Markdown("# Institutional Financial RAG Workflow Engine")
    gr.Markdown(f"**Security Limit Active:** This cloud runtime environment will strictly freeze after **{PROMPT_LIMIT}** prompts.")

    chatbot = gr.Chatbot(label="Pipeline Assistant Chat Window", height=500)
    msg_input = gr.Textbox(
        placeholder="Ask your query (e.g., 'What happened during JPMorgans (JPM) the fiscal year2026 debt instrument issuance?')", 
        label="Search Input Text Prompt"
    )
    clear_btn = gr.Button("Clear Chat")
            
    # Connect input action configurations
    msg_input.submit(
        fn=execute_gradio_pipeline,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    )
    # Clear input box after pressing enter
    msg_input.submit(lambda: "", None, msg_input)
    
    # Connect reset button configuration
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8080, theme=gr.themes.Soft())

    