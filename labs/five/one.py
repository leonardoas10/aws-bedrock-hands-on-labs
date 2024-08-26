## Invoke Bedrock model for code generation
import json
import os
import sys
import boto3
from dotenv import load_dotenv

# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

bedrock_client = boto3.client('bedrock-runtime',region_name=os.environ.get("AWS_REGION", None))

# Analyzing sales

prompt_data = """

Human: You have a CSV, sales.csv, with columns:
- date (YYYY-MM-DD)
- product_id
- price
- units_sold

Create a python program to analyze the sales data from a CSV file. The program should be able to read the data, and determine below:

- Total revenue for the year
- The product with the highest revenue
- The date with the highest revenue
- Visualize monthly sales using a bar chart

Ensure the code is syntactically correct, bug-free, optimized, not span multiple lines unnessarily, and prefer to use standard libraries. Return only python code without any surrounding text, explanation or context.

Assistant:
"""

messages=[{ "role":'user', "content":[{'type':'text','text': prompt_data}]}]

body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.5,
            "stop_sequences": ["\n\nHuman:"],
            "top_k":250
        }  
    )  

modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
response = bedrock_client.invoke_model(body=body, modelId=modelId)
response_body = json.loads(response.get('body').read())
output_list = response_body.get("content", [])
for output in output_list:
    print(output["text"])
    
## Result

"""
from collections import defaultdict
import matplotlib.pyplot as plt

# Read data from CSV file
data = []
with open('sales.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Calculate total revenue for the year
total_revenue = sum(float(row['price']) * int(row['units_sold']) for row in data)

# Find the product with the highest revenue
product_revenue = defaultdict(float)
for row in data:
    product_revenue[row['product_id']] += float(row['price']) * int(row['units_sold'])
highest_revenue_product = max(product_revenue.items(), key=lambda x: x[1])[0]

# Find the date with the highest revenue
date_revenue = defaultdict(float)
for row in data:
    date_revenue[row['date']] += float(row['price']) * int(row['units_sold'])
highest_revenue_date = max(date_revenue.items(), key=lambda x: x[1])[0]

# Visualize monthly sales
monthly_sales = defaultdict(float)
for row in data:
    month = row['date'].split('-')[1]
    monthly_sales[month] += float(row['price']) * int(row['units_sold'])

months = sorted(monthly_sales.keys())
sales = [monthly_sales[month] for month in months]

plt.figure(figsize=(10, 6))
plt.bar(months, sales)
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Monthly Sales')
plt.xticks(rotation=45)
plt.show()

print(f"Total revenue for the year: ${total_revenue:.2f}")
print(f"Product with the highest revenue: {highest_revenue_product}")
print(f"Date with the highest revenue: {highest_revenue_date}")
    
"""