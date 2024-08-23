import json
import os
import boto3
import botocore
from dotenv import load_dotenv

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

# create the prompt
prompt_data = """
Command: Write an email from Bob, Customer Service Manager, to the customer "John Doe" 
who provided negative feedback on the service provided by our customer support 
engineer"""

body = json.dumps({
    "inputText": prompt_data, 
    "textGenerationConfig":{
        "maxTokenCount":4096,
        "stopSequences":[],
        "temperature":0,
        "topP":0.9
        }
    }) 

modelId = 'amazon.titan-text-lite-v1'
accept = 'application/json'
contentType = 'application/json'
outputText = "\n"
try:

    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    outputText = response_body.get('results')[0].get('outputText')
    
    email = outputText[outputText.index('\n')+1:]
    print(email)

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error