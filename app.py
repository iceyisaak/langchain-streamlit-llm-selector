import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2']="true"
os.environ['LANGCHAIN_PROJECT']='Q&A Chatbot with OpenAI'



# Prompt Template
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please respond to the user queries"),
    ("user","Question:{question}")
])


# generate_response()
def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|output_parser
    answer=chain.invoke({"question":question})
    return answer



