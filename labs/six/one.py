## Bedrock model integration with Langchain Agents
import os
import boto3
from dotenv import load_dotenv
from langchain.agents import load_tools, Tool, AgentExecutor, create_react_agent
from langchain_aws import ChatBedrock
from langchain import LLMMathChain
from langchain_core.prompts import PromptTemplate

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

react_agent_llm = ChatBedrock(model_id=modelId, client=bedrock_client)
math_chain_llm = ChatBedrock(model_id=modelId, client=bedrock_client)
tools = load_tools(["wikipedia"], llm=react_agent_llm)

llm_math_chain =  LLMMathChain.from_llm(llm=math_chain_llm, verbose=True)

llm_math_chain.llm_chain.prompt.template = """Human: Given a question with a math problem, provide only a single line mathematical expression that solves the problem in the following format. Don't solve the expression only create a parsable expression.
```text
{{single line mathematical expression that solves the problem}}
```

Assistant:
 Here is an example response with a single line mathematical expression for solving a math problem:
```text
37593**(1/5)
```

Human: {question}

Assistant:"""

tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about math.",
    )
)

prompt_template = """Answer the following questions as best you can. 
You have access to the following tools:\n\n{tools}\n\n
Use the following format:\n\nQuestion: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\nObservation: the result of the action\n... 
(this Thought/Action/Action Input/Observation can repeat N times)\n
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question\n\nBegin!\n\n
Question: {input}\nThought:{agent_scratchpad}
"""


react_agent = create_react_agent(react_agent_llm, 
                               tools,
                                 PromptTemplate.from_template(prompt_template)
                            #    max_iteration=2,
                            #    return_intermediate_steps=True,
                            #    handle_parsing_errors=True,
                               )

question = "What is Amazon SageMaker? What is the launch year multiplied by 2"

agent_executor = AgentExecutor(
    agent=0, 
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations = 10 # useful when agent is stuck in a loop
)

agent_executor.invoke({"input": question}) 