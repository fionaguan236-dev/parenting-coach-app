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

# 2. Lock the Client into memory
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# 3. Lock the AI Chat Session into memory
if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=my_coach_rules)
    )

# 4. Create a visual memory for the chat bubbles
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Redraw all past messages every time the screen reloads
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. The Chat Input box
user_message = st.chat_input("Type your frustration here...")

if user_message:
    # Save and draw the user's message
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.chat_message("user").write(user_message)
    
    # THE UPGRADE: Draw the AI bubble, show a spinner, and catch hidden errors!
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Try to get the AI response
                response = st.session_state.chat_session.send_message(user_message)
                
                # If successful, write it and save it to memory
                st.write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                # If it fails, print the exact error in a red box on the screen
                st.error(f"Oops! A background error happened: {e}")
