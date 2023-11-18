#author @tyler @mike
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sqlite3 as sql

# change this to your LOCAL path to the CSV file. that way you use what's in the google drive @mike 
excel_file = './LabelledData_2/2.xlsx'
data = pd.read_excel(excel_file)

# connect to sql lite databse 
connection = sql.connect('smaller-train_pkts.db')

# create a cursor object - "cursor is a control structure that enables traversal over the records in a database"
cusor = connection.cursor()

# Convert the data frame to sql table
data.to_sql('train_nids_1', connection, if_exists='replace', index=False)

# read the sql table
data = pd.read_sql('SELECT * FROM train_nids_1', connection)

# commit the changes
connection.commit()
connection.close()