import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.title("Empathetic Parenting Coach 💛 (RAG Edition)")
st.write("Welcome! I am now consulting the official NVC textbook to help you.")

# --- THE QA SIDEBAR ---
# This button clears the visual chat bubbles AND the AI's hidden memory history
with st.sidebar:
    st.header("QA Testing Tools")
    st.write("Use this button to start a fresh test case and prevent token bloat!")
    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.rerun()

# 1. Initialize the API Key and the Main Brain (Back on 2.5-flash!)
api_key = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 2. Build the Knowledge Base 
@st.cache_resource(show_spinner="Reading the textbook and building the database...")
def load_knowledge_base():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever()

try:
    retriever = load_knowledge_base()
except Exception as e:
    st.error(f"Error loading the textbook: {e}")
    st.stop()

# 3. Program the Brain
system_prompt = (
    "Persona: You are an empathetic parenting coach trained in NVC and P.E.T.\n"
    "Task: Help the parent regulate emotions and guide them to an NVC I-Message.\n"
    "Strict Rules: \n"
    "1. Empathy First: Briefly validate the parent's experience in 1 sentence.\n"
    "2. THE CHECKLIST (CRITICAL): You must follow these exact steps in order. Review the Chat History. If a step is completed, YOU MUST move to the next step. DO NOT repeat steps.\n"
    "   - Step 1: Parent's Observation (What did the child do?)\n"
    "   - Step 2: Parent's Feeling (How does the parent feel?)\n"
    "   - Step 3: Parent's Need (Offer 3 multiple-choice options from the textbook)\n"
    "   - Step 4: Parent's Request (What actionable thing can the parent ask for?)\n"
    "   - Step 5: Child's Feeling & Need (Offer 3 multiple-choice options to guess the child's perspective)\n"
    "3. Progress Check: If the parent just shared their feeling, you MUST move to Step 3. If they just picked a need, you MUST move to Step 4.\n"
    "4. Ask only ONE guiding question at a time.\n"
    "\nTextbook Context:\n{context}\n"
    "\nChat History:\n{history}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Visual Memory for Chat Bubbles
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. The Chat Interface
user_message = st.chat_input("Type your frustration here...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.chat_message("user").write(user_message)
    
    with st.chat_message("assistant"):
        with st.spinner("Flipping through the textbook..."):
            try:
                chat_history_str = ""
                for msg in st.session_state.messages:
                    chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
                response = rag_chain.invoke({
                    "input": user_message,
                    "history": chat_history_str
                })
                answer = response["answer"]
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Oops! A background error happened: {e}")
