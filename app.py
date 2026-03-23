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

# 1. Initialize the API Key and the Main Brain
api_key = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 2. Build the Knowledge Base 
# THE UPGRADE: We added a custom spinner, and moved the Embeddings INSIDE the function!
@st.cache_resource(show_spinner="Reading the textbook and building the database... (This takes about 10 seconds)")
def load_knowledge_base():
    # Initialize the Translator INSIDE the cached function to prevent thread freezing
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
    
    # Load the file
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()
    
    # Chop it into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    # Translate to numbers and store in FAISS database
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever()

# Try to load the database safely
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
    "1. Empathy First: Always validate feelings.\n"
    "2. Socratic Prompting: Ask ONE guiding question per response.\n"
    "3. Linear Progression (CRITICAL): You must guide the parent sequentially through the 4 steps of NVC (1. Observation -> 2. Feeling -> 3. Need -> 4. Request). \n"
    "4. Track Progress & Never Go Backwards: If the parent has already stated their feeling, DO NOT ask for their feeling again. Move immediately to the Need. If they state their Need, move immediately to the Request.\n"
    "5. Use the attached textbook context to accurately guide them.\n"
    "\nHere is the exact textbook context to use:\n{context}"
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
                response = rag_chain.invoke({"input": user_message})
                answer = response["answer"]
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Oops! A background error happened: {e}")
