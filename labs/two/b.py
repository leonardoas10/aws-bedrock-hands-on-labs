import os
import boto3
from dotenv import load_dotenv
from langchain_aws import BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

# Config Langchain
modelId = "amazon.titan-tg1-large"
llm = BedrockLLM(
    model_id=modelId,
    model_kwargs={
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    },
    client=bedrock_client
)

shareholder_letter = "../assets/letters/2022-letter.txt"

with open(shareholder_letter, "r") as file:
    letter = file.read()
    
## this indicastes the following: Token indices sequence length is longer than the specified maximum sequence length for this model (6526 > 1024).
llm.get_num_tokens(letter)

## Split the document

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])

num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

output = ""
try:
    print("Invoke Summary Chain")
    output = summary_chain.invoke(docs)

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
    
print(output)

