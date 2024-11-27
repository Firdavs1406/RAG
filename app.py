"""
This script initializes a Streamlit app using LangChain integrations.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, "config.json")

if not os.path.exists(config_path):
    st.error("Config file not found!")
    st.stop()

with open(config_path, encoding="utf-8") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def setup_vectorstore():
    """
    Function to setup vectorstore
    """
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore


def chat_chain(vectorstore):
    """
    Function to build chain
    """
    llm = ChatGroq(model="llama-3.2-11b-vision-preview",
                   temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

    return chain


st.set_page_config(
    page_title="Dostoevskiy Chat",
    page_icon=":books:",
    layout="centered"
)

st.title("Dostoevskiy Chat :books:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstores)

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
