from collections import defaultdict
import matplotlib.pyplot as plt
import csv

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