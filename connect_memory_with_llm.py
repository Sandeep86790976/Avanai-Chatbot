import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face API token is missing! Check your .env file or environment variables.")

# Hugging Face Model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}  # Reduce response length
    )

# Custom Prompt Template (Ensures Concise Answers)
CUSTOM_PROMPT_TEMPLATE = """
Use only the provided context to answer the question. If the answer is not in the context, say "I don't know."

Context: {context}
Question: {question}

Provide a precise, structured response without additional information.
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# Get User Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Extract only the answer
answer = response["result"]

# Extract relevant source details (Page + Document Name)
source_info = []
for doc in response.get("source_documents", []):
    page = doc.metadata.get('page', 'Unknown')
    source = doc.metadata.get('source', 'Unknown Document').split("\\")[-1]  # Extract filename only
    source_info.append(f"📄 Page {page} | {source}")

# Display cleaned-up output
print("\n🔹 ANSWER:\n", answer.strip())

# Show relevant source details
if source_info:
    print("\n🔹 SOURCE DOCUMENTS:")
    for src in source_info:
        print(src)
