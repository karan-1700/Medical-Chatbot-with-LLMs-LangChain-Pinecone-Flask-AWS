
# load the pdf data
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# extract text from pdf files.
def load_pdf_files(pdf_path):
    loader = DirectoryLoader(
        path=pdf_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    # path: Path to directory.
    # glob: A glob pattern or list of glob patterns to use to find files. Defaults to "**/[!.]*" (all files except hidden).
    # loader_cls: Loader class to use for loading files. Defaults to UnstructuredFileLoader.

    documents = loader.load()
    return documents


from typing import List
from langchain.schema import Document
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# split the documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # consider 500 characters as 1 chunk.
        chunk_overlap=20, # overlap 20 characters for each chunk.
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


# get the embeddings model.
# Embedding models are machine learning models that transform data 
# (like text, images, or audio) into numerical representations called embeddings. 
# These embeddings capture the semantic meaning and relationships within the data, 
# allowing machines to understand and compare them in a meaningful way.
from langchain.embeddings import HuggingFaceEmbeddings
def download_mebeddings():
    """
    Download and return the HuggingFace embeddings model.

    Here, I am using the below embedding model:
    all-MiniLM-L6-v2
    This is a sentence-transformers model: It maps sentences & paragraphs to a 
    384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings



# # connect the LLM
# from langchain_openai import ChatOpenAI
# chat_model = ChatOpenAI(model="@cf/openai/gpt-oss-120b")

from langchain_core.language_models import LLM
from typing import Optional, List
import requests
import json

class CloudflareLLM(LLM):
    cloudflare_user_id: str
    api_key: str
    model: str = "@cf/openai/gpt-oss-120b"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_user_id}/ai/run/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({
            "input": prompt
        })
        response = requests.post(url, headers=headers, data=payload)
        result = response.json()

        if not result["success"]:
            raise Exception(result["errors"])
        
        return result["result"]["output"][-1]["content"][0]["text"]  # This depends on Cloudflare's exact response structure

    @property
    def _llm_type(self) -> str:
        return "cloudflare_llm"