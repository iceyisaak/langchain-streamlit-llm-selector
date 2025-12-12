import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# streamlit run app_openai.py

import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OpenAI"



# Prompt Template
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please respond to the user queries"),
    ("user","Question:{question}")
])


# generate_response()
def generate_response(question, api_key, llm_name, temperature, max_tokens):
    """
    Initializes ChatOpenAI with all parameters and invokes the chain.
    """
    # 1. Initialize ChatOpenAI with ALL necessary parameters
    # The 'openai_api_key' MUST be passed here, along with temperature and max_tokens.
    llm = ChatOpenAI(
        openai_api_key=api_key, # CRITICAL FIX: Pass the key here
        model=llm_name,         # Use the model name passed (e.g., "gpt-4o")
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # 2. Define the Output Parser
    output_parser = StrOutputParser()
    
    # 3. Create the Chain (Prompt | LLM | Output Parser)
    chain = prompt | llm | output_parser
    
    # 4. Invoke the chain
    answer = chain.invoke({"question": question})
    
    return answer


################################### Streamlit UI #########################################

# Title of App
st.title("Q&A Chatbot with OpenAI")
api_key=st.sidebar.text_input("Enter your OpenAI API Key", type='password')


# Dropdown Menu
llm_name=st.sidebar.selectbox("Select an OpenAI LLM model",["gpt-4o","gpt-4-turbo","gpt-4"])


# Parameter Adjustments
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


### Main UI

st.write("Please feel free to ask me any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm_name,temperature,max_tokens)
    st.write(response)
else:
    st.write("")