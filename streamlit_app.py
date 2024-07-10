import streamlit as st
from openai import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI as LangChainOpenAI
import os

# Show title and description
st.title("📄 Document Question Answering with Web Search")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide OpenAI and Serper API keys."
)

# Ask user for API keys
openai_api_key = st.text_input("OpenAI API Key", type="password")
serper_api_key = st.text_input("Serper API Key", type="password")

if not openai_api_key or not serper_api_key:
    st.info("Please add your OpenAI and Serper API keys to continue.", icon="🗝️")
else:
    # Set environment variables for API keys
    os.environ["SERPER_API_KEY"] = serper_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Create OpenAI clients
    client = OpenAI(api_key=openai_api_key)
    llm = LangChainOpenAI(temperature=0, openai_api_key=openai_api_key)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Question input
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API
        st.write("### GPT Answer")
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

        # Web search functionality
        st.write("### Web Search Result")
        with st.spinner("Searching the web..."):
            try:
                search = GoogleSerperAPIWrapper()
                tools = [
                    Tool(
                        name="Intermediate Answer",
                        func=search.run,
                        description="useful for when you need to answer questions that require up-to-date information or additional context"
                    )
                ]

                agent = initialize_agent(
                    tools, 
                    llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                    verbose=True
                )
                
                search_result = agent.run(f"Based on the following question, provide relevant information from a web search: {question}")
                st.write(search_result)
            except Exception as e:
                st.error(f"An error occurred during web search: {str(e)}")
