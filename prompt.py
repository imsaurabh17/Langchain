from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
You are an expert research paper summarizer.

The user will provide {user_input} which is a reaserch paper name and

TASK:
Generate a summary of the research paper strictly following the explanation_type and length.

The explanation type will be {explanation_type} and lenth will be {length} 

RESTRICTIONS:
- Don’t invent content not in the paper.
- Don’t include long quotes.
- Output only the final summary.
""",
input_variables=["user_input","explanation_type","length"]
)

template.save("template.json")
