import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import pandas as pd
import altair as alt
import os
import time
import base64
import io

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google Serper
search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load configuration file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create an authentication object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def process_document(document, question):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(document)
    
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-turbo")
    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for finding additional information online"
        )
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    final_answer = ""
    for i, chunk in enumerate(chunks):
        prompt = f"""
        Analyze the following document chunk and answer the question. If you need additional information, 
        use the web search tool provided. This is chunk {i+1} of {len(chunks)}.

        Document Chunk:
        {chunk}

        Question: {question}

        Provide a clear and concise answer based on the document content and any additional 
        information from web search if necessary.
        """
        chunk_result = agent.run(prompt)
        final_answer += chunk_result + "\n\n"
    
    return final_answer

def main():
    st.title("📄 Advanced Document Q&A with Web Search")

    # Authentication
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        st.write(f'Welcome *{name}*')
        
        uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))
        
        if uploaded_file:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the maximum limit of {MAX_FILE_SIZE/1024/1024:.2f} MB.")
            else:
                document = uploaded_file.read().decode()
                st.success("File uploaded successfully!")

                # Document analysis
                with st.expander("Document Analysis"):
                    entities = perform_ner(document)
                    st.write("### Named Entities")
                    st.write(entities)

                    # Word cloud
                    words = document.split()
                    word_freq = pd.DataFrame(pd.Series(words).value_counts()).reset_index()
                    word_freq.columns = ['word', 'count']
                    
                    chart = alt.Chart(word_freq.head(50)).mark_circle().encode(
                        size='count',
                        color=alt.value('steelblue'),
                        tooltip=['word', 'count']
                    ).interactive()
                    
                    st.write("### Document Word Cloud")
                    st.altair_chart(chart, use_container_width=True)

                # Q&A
                question = st.text_area(
                    "Ask a question about the document",
                    placeholder="Can you give me a short summary?",
                    max_chars=500
                )

                if question:
                    with st.spinner("Processing your question..."):
                        try:
                            answer = process_document(document, question)
                            st.write("### Answer")
                            st.write(answer)

                            # Add to history
                            if 'history' not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append({"question": question, "answer": answer})

                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

                # Display history
                if 'history' in st.session_state and st.session_state.history:
                    with st.expander("Question History"):
                        for item in st.session_state.history:
                            st.write(f"Q: {item['question']}")
                            st.write(f"A: {item['answer']}")
                            st.write("---")

        # Logout button
        authenticator.logout('Logout', 'main')

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()
