# storing vector in the knowledge base.
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_mebeddings

# load the Pinecone API key
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# laod the documents.
extracted_pdf_data = load_pdf_files("data")
print("[INFO] Loaded documents from pdf file(s)...")
# filter the documents.
minimal_docs = filter_to_minimal_docs(extracted_pdf_data)
# print(len(minimal_docs))
# chunking the documents.
text_chunks = text_split(minimal_docs=minimal_docs)
print(f"Number of chunks: {len(text_chunks)}")
print("[INFO] chunking the documents...")

# download the hugging face embeddings model.
embedding_model = download_mebeddings()
print("[INFO] downloaded a vector embedding model from hugging face...")

# create an index in Pinecone.
from pinecone import Pinecone
# initialize a pinecone client with your API key.
pc = Pinecone(api_key=PINECONE_API_KEY)
print("[INFO] initialized Pinecone...")

# create a database 
# Create a dense index with integrated embedding
from pinecone import ServerlessSpec

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        metric="cosine", # consine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=384 # dimension of the embeddings
    )

index = pc.Index(index_name)

# take all of the text chunks, use the embedding model to convert the chunks 
# into vector embedding, and store it in Pinecone vector database.
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, 
    embedding=embedding_model, 
    index_name=index_name
)
print("[INFO] converted to vector embeddings and uploaded to pinecone vector database...")


# # Once we are done storing the vectors in pinecone vector database, 
# # we usually want to:
# #     Run similarity search
# #     Query existing data
# #     Use retrieval for a chatbot or RAG pipeline
# # To do that, we do not need to recreate the index from documents — 
# # instead, we just connect to it using from_existing_index()
# from langchain_pinecone import PineconeVectorStore
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embedding_model
# )