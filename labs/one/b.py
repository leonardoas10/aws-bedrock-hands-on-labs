import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock as Bedrock
from langchain.prompts import PromptTemplate

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)


bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

inference_modifier = { 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }

chat = Bedrock(model_id = modelId, model_kwargs = inference_modifier)

# Create a prompt template
prompt_template = PromptTemplate(template="Human: {text}")

# Format the prompt with specific input
formatted_prompt = prompt_template.format(text="Hello!")

hello = chat.invoke(formatted_prompt)
print(hello.content)

# Create a prompt template that has multiple input variables
multi_var_prompt = PromptTemplate(
    input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"], 
    template="""

Human: Create an apology email from the Service Manager {customerServiceManager} to {customerName} in response to the following feedback that was received from the customer: 
<customer_feedback>
{feedbackFromCustomer}
</customer_feedback>

Assistant:"""
)

# Pass in values to the input variables
prompt = multi_var_prompt.format(customerServiceManager="Bob", 
                                 customerName="John Doe", 
                                 feedbackFromCustomer="""Hello Bob,
     I am very disappointed with the recent experience I had when I called your customer support.
     I was expecting an immediate call back but it took three days for us to get a call back.
     The first suggestion to fix the problem was incorrect. Ultimately the problem was fixed after three days.
     We are very unhappy with the response provided and may consider taking our business elsewhere.
     """
     )

num_tokens = chat.get_num_tokens(prompt)
print(f"Our prompt has {num_tokens} tokens")

response = chat.invoke(prompt)

print(response.content)