import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

# Streamlit app
st.title('LangChain Text Summarizer')

openai_api_key = st.text_input("OpenAI API Key", type="password")
source_text = st.text_area("Source Text", height=200)

if st.button("Summarize"):
    if not openai_api_key.strip() == "" and not source_text.strip() == "":
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(source_text)

        docs = [Document(page_content=t) for t in texts[:3]]

        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        
        st.write(summary)
    else:
        st.write(f"Please complete the missing fields.")
