import streamlit as st
from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import os

# Show title and description.
st.title("📄 Document Question Answering with Web Search")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
serper_api_key = st.text_input("Serper API Key", type="password")

if not openai_api_key or not serper_api_key:
    st.info("Please add your OpenAI and Serper API keys to continue.", icon="🗝️")
else:
    # Set environment variables for API keys
    os.environ["SERPER_API_KEY"] = serper_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question.
        document = uploaded_file.read().decode()

        prompt_template = f"""
You are an AI assistant designed to answer questions about uploaded documents. Your task is to process a given document, understand its content, and then answer a specific question about it. You will also need to use provided API keys for additional functionality if necessary.

First, you will receive the content of an uploaded document:

<document>
{document}
</document>

Next, you will be given a question about this document:

<question>
{question}
</question>

To process the document and answer the question, follow these steps:

1. Carefully read and analyze the content of the document.
2. Identify key information, main ideas, and relevant details that relate to the question.
3. If the question can be answered directly from the document content, formulate a clear and concise answer based on the information provided.
4. If additional information is needed to fully answer the question, you may use the provided API keys to access external resources. The API keys are:

   OpenAI API Key: {openai_api_key}
   Serper API Key: {serper_api_key}

   Use these keys responsibly and only when necessary to supplement the information in the document.

5. When using external resources, clearly indicate in your answer that additional information was used and briefly explain why it was necessary.

6. Ensure your answer is comprehensive, accurate, and directly addresses the question asked.

7. If the question cannot be answered based on the document content and available resources, clearly state this and explain why.

Format your response as follows:

<answer>
[Your detailed answer to the question, based on the document and any additional information if used]
</answer>

<sources>
[If external resources were used, list them here with brief descriptions of how they contributed to the answer]
</sources>

Remember to maintain a professional and helpful tone throughout your response. If you need any clarification or additional information to complete this task, please ask.
"""

          # Generate an answer using the OpenAI API.
    messages = [
        {
            "role": "user",
            "content": prompt_template,
        }
    ]

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write`.
    response = st.empty()
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            response.markdown(full_response)

    # Add web search functionality
    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="Useful for when you need to answer with search"
        )
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    search_result = self_ask_with_search.run(question)

    st.write("### Web Search Result")
    st.write(search_result)
