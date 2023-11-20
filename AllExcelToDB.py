#author @tyler @mike
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sqlite3 as sql


excel_files = ['1.xlsx', '2.xlsx', '3.xlsx', '4.xlsx',]

# make an empty dataframe
combined_data = pd.DataFrame()


# loop through all the excel files

for i, file in enumerate(excel_files):
    
    excel_file = f'./LabelledData_2/{file}' # this gets the file path for each file (1.xlsx, 2.xlsx, etc)
    
    # read the excel file, skip the first row (they are headers), and do not skip for the first file.
    skip_rows = 0 if i == 0 else 1
    data = pd.read_excel(excel_file, skiprows=skip_rows, header=0)
    
    
    #  concat the data to the combined data frame
    combined_data = pd.concat([combined_data, data], axis=0, ignore_index=True)
    
    
# connect to sql lite database, create cursor object, make sql, read sql
connection = sql.connect('LabelledData_2.db')
cursor = connection.cursor()
combined_data.to_sql('combined_2', connection, if_exists='replace', index=False)
data_from_db = pd.read_sql('SELECT * FROM combined_2', connection)

# commit changes and close connection
connection.commit()
connection.close()
