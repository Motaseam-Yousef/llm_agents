import streamlit as st
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize knowledge base and storage
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url)
)

knowledge_base.load()

storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

# Streamlit App
st.set_page_config(page_title="PDF Assistant", page_icon="🔍", layout="wide")
st.title("📘 PDF Assistant")
st.markdown("""
### Chat with your PDF Assistant!
This assistant is powered by the knowledge base extracted from the document:
[Thai Recipes PDF](https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf). 
Type your queries to interact with the assistant and get relevant results from the knowledge base.
""")

def pdf_assistant(new: bool, user: str):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        st.success(f"Started New Run: {run_id}")
    else:
        st.info(f"Continuing Existing Run: {run_id}")

    return assistant

# Sidebar Configuration
st.sidebar.header("Configuration")
new_run = st.sidebar.checkbox("Start New Run", value=False)
user_id = st.sidebar.text_input("User ID", value="user")

# Initialize Assistant
assistant = pdf_assistant(new=new_run, user=user_id)

# Chat Interface
st.header("🗨️ Chat Interface")
user_input = st.text_area("Ask your question:", key="user_input")
if st.button("Send"):
    if user_input.strip():
        try:
            # Fetch the response from the assistant
            response = assistant.chat(user_input)
            response_text = "".join([str(chunk) for chunk in response])
            st.markdown(f"**Assistant:** {response_text}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a message to send.")
