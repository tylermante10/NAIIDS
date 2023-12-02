import pandas as pd

# Read Excel file
df = pd.read_excel('./LabelledData_2/3.xlsx') # Set 2 Mike
#df = pd.read_table('./LabelledData_3/4.xlsx') # Set 3 Joe


# Print the first row
first_row = df.head(n=1)
print(df)
