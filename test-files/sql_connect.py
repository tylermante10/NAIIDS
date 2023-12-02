#author @tyler @mike
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sqlite3 as sql
# HERE: FLIP THE COMMENTS
# change this to your LOCAL path to the CSV file. that way you use what's in the google drive @mike 
# excel_file = './UNSW-NB15_Labelled_Set_2.xlsx'
excel_file = '/mnt/c/Users/mante/Downloads/UNSW-NB15_Labelled_Set_2.xlsx'



# read the excel file
data = pd.read_excel(excel_file)

# connect to sql lite databse 
connection = sql.connect('test.db')

# create a cursor object - "cursor is a control structure that enables traversal over the records in a database"
cusor = connection.cursor()

# Convert the data frame to sql table
data.to_sql('3rd-set', connection, if_exists='replace', index=False)

# read the sql table
data = pd.read_sql('SELECT * FROM test_1', connection)

# commit the changes
connection.commit()
connection.close()