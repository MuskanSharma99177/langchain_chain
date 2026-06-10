from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# OpenAI Model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

parser = StrOutputParser()

# Output Schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Classification Prompt
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback text into positive or negative.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

# Positive Response Prompt
prompt2 = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Negative Response Prompt
prompt3 = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Branching Logic
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive",
     prompt2 | model | parser),

    (lambda x: x.sentiment == "negative",
     prompt3 | model | parser),

    RunnableLambda(lambda x: "Could not determine sentiment")
)

# Complete Chain
chain = classifier_chain | branch_chain

result = chain.invoke({
    "feedback": "This is a beautiful phone"
})

print(result)

chain.get_graph().print_ascii()