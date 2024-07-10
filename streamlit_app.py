import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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

def process_documents(documents, question):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    # Note: Replace 'your-openai-api-key' with your actual OpenAI API key
    llm = OpenAI(temperature=0, openai_api_key='your-openai-api-key')
    agent = initialize_agent([], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    final_answer = ""
    for i, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc)
        for j, chunk in enumerate(chunks):
            prompt = f"""
            Analyze the following document chunk and answer the question. 
            This is document {i+1} of {len(documents)}, chunk {j+1} of {len(chunks)}.

            Document Chunk:
            {chunk}

            Question: {question}

            Provide a clear and concise answer based on the document content.
            """
            chunk_result = agent.run(prompt)
            final_answer += f"Document {i+1}, Chunk {j+1}:\n{chunk_result}\n\n"
    
    return final_answer

def compare_documents(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    common_words = vectorizer.get_feature_names_out()
    tfidf_sums = np.sum(tfidf_matrix.toarray(), axis=0)
    top_words = sorted(zip(common_words, tfidf_sums), key=lambda x: x[1], reverse=True)[:20]
    
    return cosine_sim, top_words

def visualize_similarity(cosine_sim):
    similarity_df = pd.DataFrame(cosine_sim)
    similarity_df.index = [f"Doc {i+1}" for i in range(len(cosine_sim))]
    similarity_df.columns = [f"Doc {i+1}" for i in range(len(cosine_sim))]
    
    heatmap = alt.Chart(similarity_df.reset_index().melt(id_vars='index')).mark_rect().encode(
        x='index:O',
        y='variable:O',
        color='value:Q',
        tooltip=['index', 'variable', 'value']
    ).properties(
        title="Document Similarity Heatmap",
        width=400,
        height=400
    )
    
    return heatmap

def main():
    st.title("📄 Multi-Document Comparison and QA")

    uploaded_files = st.file_uploader("Upload documents (.txt or .md)", type=("txt", "md"), accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            if file.size > MAX_FILE_SIZE:
                st.error(f"File {file.name} exceeds the maximum limit of {MAX_FILE_SIZE/1024/1024:.2f} MB.")
            else:
                documents.append(file.read().decode())
        
        st.success(f"{len(documents)} files uploaded successfully!")

        # Document comparison
        if len(documents) > 1:
            st.subheader("Document Comparison")
            cosine_sim, top_words = compare_documents(documents)
            
            st.write("### Document Similarity")
            similarity_chart = visualize_similarity(cosine_sim)
            st.altair_chart(similarity_chart, use_container_width=True)
            
            st.write("### Top Common Words")
            word_chart = alt.Chart(pd.DataFrame(top_words, columns=['word', 'tfidf'])).mark_bar().encode(
                x='tfidf:Q',
                y=alt.Y('word:N', sort='-x'),
                tooltip=['word', 'tfidf']
            ).properties(
                title="Top 20 Common Words Across Documents",
                width=600,
                height=400
            )
            st.altair_chart(word_chart, use_container_width=True)

        # Document analysis
        for i, doc in enumerate(documents):
            with st.expander(f"Document {i+1} Analysis"):
                entities = perform_ner(doc)
                st.write("### Named Entities")
                st.write(entities)

                # Word cloud
                words = doc.split()
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
            "Ask a question about the document(s)",
            placeholder="What are the main themes across all documents?",
            max_chars=500
        )

        if question:
            with st.spinner("Processing your question..."):
                try:
                    answer = process_documents(documents, question)
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

if __name__ == "__main__":
    main()
