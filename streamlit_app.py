import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

# Streamlit app
st.title('LangChain Text Summarizer')

# Get OpenAI API key and source text input
openai_api_key = st.text_input("OpenAI API Key", type="password")
source_text = st.text_area("Source Text", height=200)

# Check if the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_text.strip():
        st.write(f"Please complete the missing fields.")
    else:
        try:
            # Split the source text
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(source_text)

            # Create Document objects for the texts
            docs = [Document(page_content=t) for t in texts[:3]]

            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)

            # Display summary
            st.write(summary)
        except Exception as e:
            st.write(f"An error occurred: {e}")
