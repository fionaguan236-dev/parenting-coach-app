import streamlit as st
from google import genai
from google.genai import types

st.title("Empathetic Parenting Coach 💛")
st.write("Welcome! I am here to help you navigate frustrating parenting moments.")

# 1. The Brain
my_coach_rules = """
Persona: You are an empathetic parenting coach trained in NVC and P.E.T.
Task: Help the parent regulate emotions and guide them to an NVC I-Message.
Rules: 
1. Empathy First. 
2. Socratic Prompting: Ask ONE guiding question at a time.
3. Provide 2-3 psychological hints before asking them to guess the child's need.
"""

# 2. THE FIX: Lock the Client (Waiter) into Streamlit's Memory
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# 3. Lock the Chat Session into Streamlit's Memory
if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=my_coach_rules)
    )

# 4. The Chat Interface
user_message = st.chat_input("Type your frustration here...")

if user_message:
    st.chat_message("user").write(user_message)
    
    # Send the message using the Chat Session we saved in memory
    response = st.session_state.chat_session.send_message(user_message)
    st.chat_message("ai").write(response.text)
# 5. The Chat Interface
user_message = st.chat_input("Type your frustration here...")

if user_message:
    # Show what the user typed on the screen
    st.chat_message("user").write(user_message)
    
    # Send it to Gemini and show the AI's response
    response = st.session_state.chat_session.send_message(user_message)
    st.chat_message("ai").write(response.text)
