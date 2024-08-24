## Chatbot - Contextual Aware
import os
import boto3
from dotenv import load_dotenv
from langchain_aws import BedrockLLM
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Initialize Bedrock clients
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", None))
ai21_llm = BedrockLLM(model_id="ai21.j2-ultra-v1", client=bedrock_client, model_kwargs={
    "maxTokens": 4096,
    "stopSequences": [],
    "temperature": 0,
    "topP": 1,
})
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Load and process documents
loader = CSVLoader("../assets/rag_data/Amazon_SageMaker_FAQs.csv")
documents_aws = loader.load()
print(f"Documents loaded: {len(documents_aws)}")

docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)
print(f"Documents after splitting and chunking: {len(docs)}")

# Create FAISS vector store
try:
    vectorstore_faiss_aws = FAISS.from_documents(
        documents=docs,
        embedding=br_embeddings,
    )
    print("Vector store created successfully.")
except ValueError as error:
    if "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\n"
              "To troubleshoot this issue please refer to the following resources:\n"
              "https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\n"
              "https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        raise SystemExit
    else:
        raise error

# Create the VectorStoreIndexWrapper
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)

# Define the QA chain
memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=ai21_llm, 
    retriever=vectorstore_faiss_aws.as_retriever(),
    memory=memory_chain,
    chain_type='stuff',
)

# Customize the prompt template
qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
{context}:

Use at maximum 3 sentences to answer the question. 

{question}:

If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."

Answer:""")

# Terminal-based chat loop
def start_chat():
    print("Starting chat bot. Type 'q' or 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {'q', 'quit'}:
            print("Thank you, that was a nice chat!")
            break
        if user_input:
            try:
                result = qa({"question": user_input})
                print(f"AI: {result['answer']}\n")
            except Exception as e:
                print(f"Error: {str(e)}")

# Start the chat
start_chat()