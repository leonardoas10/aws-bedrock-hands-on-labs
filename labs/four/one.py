## Chatbot (Basic - without context)
import warnings
warnings.filterwarnings('ignore')
import json
import os
import sys
import boto3
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_aws import BedrockLLM
from langchain.memory import ConversationBufferMemory

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

ai21_llm = BedrockLLM(model_id="ai21.j2-ultra-v1", client=bedrock_client, model_kwargs={
        "maxTokens": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    })
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ai21_llm, verbose=True, memory=memory
)

try:
    
    print(conversation.predict(input="Hi there!"))
    print(conversation.predict(input="Give me a few tips on how to start a new garden."))
    print(conversation.predict(input="Vegetable"))
    print(conversation.predict(input="That's all, thank you!"))

except ValueError as error:
    if  "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error