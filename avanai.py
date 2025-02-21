import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def load_llm():
    """Load LLM from Hugging Face API."""
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

def set_custom_prompt():
    """Custom prompt for better structured answers."""
    CUSTOM_PROMPT_TEMPLATE = """
    Use only the provided context to answer the question.  
    If the answer is not in the context, say "I don't know."  
    Do NOT guess or provide extra information.

    Context: {context}  
    Question: {question}  

    Provide a clear and structured response.
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def main():
    """Streamlit UI for Avanai Chatbot."""
    st.title("Avanai: University Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask Avanai...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),  # Increased k for accuracy
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})

            # Extract answer & sources
            answer = response["result"]
            sources = response["source_documents"]

            # Show only relevant source details
            source_info = []
            for doc in sources:
                page = doc.metadata.get('page', 'Unknown')
                source = doc.metadata.get('source', 'Unknown Document').split("\\")[-1]
                source_info.append(f"📄 Page {page} | {source}")

            # Display final response
            formatted_response = f"**📢 Answer:**\n{answer}\n\n"
            if source_info:
                formatted_response += "**📌 Sources:**\n" + "\n".join(source_info)

            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
