from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

st.header("Reaserch assistant")

user = st.text_input("Enter your prompt: ")

if st.button("Generate Summary"):
    result = model.invoke(user)
    st.write(result.content)