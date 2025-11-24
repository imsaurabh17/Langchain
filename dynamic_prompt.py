from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.header("Research Assistant")

user_input = st.selectbox("Topics",
                           options=[
                            "Attention Is All You Need",
                            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                            "ResNet: Deep Residual Learning for Image Recognition",
                            "U-Net: Convolutional Networks for Biomedical Image Segmentation",
                            "Denoising Diffusion Probabilistic Models (DDPM)"
                           ])

explanation_type = st.selectbox("Explanation Type",
                                options=[
                                    "code-heavy → use pseudocode or code snippets",
                                    "math-heavy → equations, formulas, and derivations",
                                    "theory-focused → focus on intuition and concepts ",
                                    "example-driven → use analogies",
                                    "beginner-friendly → simplify terms, avoid heavy math",
                                    "interview-style → bullet points, crisp takeaways"
                                ])

length = st.selectbox("Length of the explanation",
                      options=[
                          "very-short → 3–4 bullet points or sentences",
                          "short → 1–2 paragraphs",
                          "medium → 3–5 paragraphs",
                          "long → multi-section detailed summary"
                      ])

prompt = load_prompt("template.json")

prompt = prompt.format(
    user_input=user_input,
    explanation_type=explanation_type,
    length = length
)

model = ChatGroq(model="openai/gpt-oss-120b")

if st.button("Generate summary"):
    result = model.invoke(prompt)
    st.write(result.content)
