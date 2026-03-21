import streamlit as st
from google import genai
from google.genai import types

# 1. Set up the visual page
st.title("Empathetic Parenting Coach 💛")
st.write("Welcome! I am here to help you navigate frustrating parenting moments.")

# 2. Get the secure key from Streamlit's vault
api_key = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

# 3. The Brain (System Instructions)
my_coach_rules = """
Persona: You are an empathetic parenting coach trained in NVC and P.E.T.
Task: Help the parent regulate emotions and guide them to an NVC I-Message.
Rules: 
1. Empathy First. 
2. Socratic Prompting: Ask ONE guiding question at a time.
3. Provide 2-3 psychological hints before asking them to guess the child's need.
"""

# 4. Streamlit Memory (Websites reload every time you type, so we must save the chat history)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=my_coach_rules)
    )

# 5. The Chat Interface
user_message = st.chat_input("Type your frustration here...")

if user_message:
    # Show what the user typed on the screen
    st.chat_message("user").write(user_message)
    
    # Send it to Gemini and show the AI's response
    response = st.session_state.chat_session.send_message(user_message)
    st.chat_message("ai").write(response.text)
