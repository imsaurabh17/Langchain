from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, List
from dotenv import load_dotenv

load_dotenv()

class Extract(BaseModel):
    name : str = Field(description="Extract the name of the candidate")
    mobile_no : str = Field(description="Extract a 10 digit mobile number")
    email : Annotated[str, "Extract the email id of the candidate"]
    skills : Annotated[List[str],"Extract the list of skills from the resume provided"]
    experience: Annotated[str, "Extract the experience of the candidate"]
    summary: Annotated[str,"Give a brief summary about the candidate"]
    suggestion: Annotated[str, "Provide some useful suggestion about the candidate"]

parser = PydanticOutputParser(pydantic_object=Extract)

model = ChatGroq(model="openai/gpt-oss-120b")

template = PromptTemplate(
    template = "Provide the info about the {candidate} \n {format_instructions}",
    input_variables=['candidate'],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"candidate":"""
RAMESH K
Email : rameshkr.123@@gmail..com
Phone- 98 76 54 321
Bangalore / sometimes in Chennai
LinkedIn : (not updated)

Objective / Summary / Something like that??

To get a gud job in IT where I can use my skills (Python?? SQL??) and learn new things. I have done many projects in college and internships etc. Hardworking. Quick learner. Team Player.

SKILLS (not sure exact)

python, PYTHON(basic), excel, ML?? (just basics),
sql (medium),
tableau??? (used once)
communication maybe
C, C++, HTML-css little bit

EXPERIENCE (kind of)

Intern @ ABC Comp
2023 (2 months?)

Worked on some data cleaning stuff

Made an Excel report for sales or finance (not sure)

Helped team in documentation

Freelance??
2022 – 2024 sometimes
Did some small python scripts for people (file renaming, whatsapp automation, etc)

EDUCATION

Btech CSE – 2020-2024 – College of Engineering (tier 3)
12th – 2019 – some school in Karnataka
10th – 2017 – same school

PROJECTS (not arranged properly)

Face detection system using python + cv2
Worked sometimes, but accuracy was low. Detected faces.
Repo lost.

something with chatbot
Made a rule based chatbot using python. Only basic replies.

ML model for marks prediction
Used linear regression. Accuracy approx 76%.
Dataset created manually from friends.

Certifications and Stuff

Python for Everybody (didn’t finish all modules)

NPTEL course on Data Analytics – scored 47/100 (pass?? unsure)

Java course on Udemy. (did 40% only)

OTHER

Languages: English ok-ok, Hindi good, Kannada beginner
Hobbies = cricket, mobile games, netflix
Sometimes volunteer in college fest

"""})

print(result)