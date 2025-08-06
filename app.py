
from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from src.helper import download_mebeddings
from src.prompt import *
from src.helper import CloudflareLLM
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_ID = os.getenv("USER_ID")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# get the embeddings model.
embedding_model = download_mebeddings()

# Once we are done storing the vectors in pinecone vector database, 
# we usually want to:
#     Run similarity search
#     Query existing data
#     Use retrieval for a chatbot or RAG pipeline
# To do that, we do not need to recreate the index from documents — 
# instead, we just connect to it using from_existing_index()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

# now, we have stored everything in our knowledge: pinenone vector database.
# now, we will create a retriever, and connect the LLM.
retriever = docsearch.as_retriever(search_type="similarity", 
                                   search_kwargs={"k" : 3}) # get 3 most similar responses from the vector dataset.


# # connect the LLM
llm = CloudflareLLM(
    cloudflare_user_id=USER_ID,
    api_key=OPENAI_API_KEY,
    model="@cf/openai/gpt-oss-120b"
)


# create a langchain chain.
# Create a chat prompt template from a variety of message formats.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create a chain for passing a list of Documents to a model.
question_answer_chain = create_stuff_documents_chain(llm=llm, 
                                                     prompt=prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# create a default route
@app.route("/")
def index():
    return render_template('chat.html')

# whenver the user clicks on the send button in the UL, this route will get executed.
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"] # use .get to avoid KeyError
    # print("Input received:", msg)
    if not msg:
        return "No message received", 400
    
    response = rag_chain.invoke({"input" : msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
