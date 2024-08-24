## Chatbot using prompt
import os
import boto3
from dotenv import load_dotenv
from langchain_aws import BedrockLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", None))

ai21_llm = BedrockLLM(model_id="ai21.j2-ultra-v1", client=bedrock_client, model_kwargs={
    "maxTokens": 4096,
    "stopSequences": [],
    "temperature": 0,
    "topP": 1,
})

qa = ConversationChain(
    llm=ai21_llm, verbose=False, memory=ConversationBufferMemory()
)

def start_chat():
    print("Starting chat bot. Type 'q' or 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {'q', 'quit'}:
            print("Thank you, that was a nice chat!")
            break
        if user_input:
            try:
                result = qa.invoke({'input': user_input})
                print(f"AI: {result}\n")
            except Exception as e:
                print(f"Error: {str(e)}")

# Start the chat
start_chat()