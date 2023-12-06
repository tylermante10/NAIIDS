# testing an SGD classifier
# author @tyler @mike

# do the imports
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sqlite3 as sql
import numpy as np
import gc

# create connection - no need to adjust path
connection = sql.connect('train.db')

# create cursor
cursor = connection.cursor()

# this array is a 2D array, on sprint planning this is 3.d.i
# we want every column besides flag and label.
# List of columns to include
included_columns = ['Src_ip_A', 'Src_ip_B', 'Src_ip_C', 'Src_ip_D', 'Source_Port', 'Protocol_feature',
                    'State_feature', 'Row_dur', 'src_bytes', 'dest_bytes', 'Src_ttl', 'Dest_ttl', 'Src_loss', 'Dest_loss',
                    'Service_feature', 'Src_bps', 'Dest_bps', 'Src_pkt_ct', 'Dst_pkt_ct', 'Src_win_val', 'Dst_win_val',
                    'Src_tcp_bsn', 'Dst_tcp_bsn', 'Src_mean_pkt_size', 'Dst_mean_pkt_size', 'Trans_depth', 'Resp_bdy_len',
                    'Src_jitter', 'Dest_jitter', '"Start Time"', '"Last Time"', 'Src_arr_time', 'Dest_arr_time', 'Tcp_rtt',
                    'Synack_time', 'Ack_time', 'Is_sm_ips_ports', 'Ct_state_ttl', 'Ct_http_f', 'login_ftp', 'Ct_ftp_cmd',
                    'Ct_srv_src', 'Ct_srv_dest', 'Ct_dest_ltm', 'Ct_str_ltm', 'Ct_src_dport_ltm', 'Ct_dest_sport_ltm',
                    'Ct_dest_src_ltm', 'Dest_ip_A', 'Dest_ip_B', 'Dest_ip_C', 'Dest_ip_D', 'Dest_Port']

# Combine column names into a single string
SQL_select = ', '.join(included_columns)

# seems to be some edge case where it gets rows that are filled with None? so have it not include those
execute = f"SELECT {SQL_select} FROM sample_train_1;"
cursor.execute(execute)

# get the result of that execute. this will be the 2D array with every column except flag
result_features = cursor.fetchall()
print("Features selection success")

# let's repeat but for the flag column (or label, we can easily switch it out)
execute = f"SELECT Flag FROM sample_train_1;"
cursor.execute(execute)

result_label = cursor.fetchall()
print("Training label selection success")
## Convert results to arrays

# this array is a 2D array if sample number and feature, on sprint planning this is 3.d.i
features_train = np.array(result_features, dtype='float32')

# this array is a 1d array of labels, on sprint planning this is 3.d.ii
labels_train = np.array(result_label, dtype='float32')
labels_train = labels_train.flatten()
print("Features and labels successfully converted to numpy arrays")

# close connection and cursor
cursor.close()
connection.close()


### Testing Section
# now reopen the connection and get features and labels for testing data - no need to adjust path
connection = sql.connect('test.db')
cursor = connection.cursor()

# follow same procedure as above
# this is a 2D array, all columns excluding flag 
execute = f"SELECT {SQL_select} FROM sample_test_1;"
cursor.execute(execute)

# fetch reults and create testing 2D array
result_features = cursor.fetchall()
features_test = np.array(result_features, dtype='float32')

# this is a 1D array, only flag column
execute = f"SELECT Flag FROM sample_test_1;"
cursor.execute(execute)

# fetch results and create testing 1D array
result_label = cursor.fetchall()
labels_test = np.array(result_label, dtype='float32')
labels_test = labels_test.flatten()


# machine learn model object
clf = SGDClassifier() #default parameters

# train the data
print("Training...")
clf.fit(features_train, labels_train.ravel()) 
print("Training complete")

print("\n")
# make predictions
labels_pred = clf.predict(features_test)

# # get accuracy
accuracy = clf.score(features_test, labels_test)
accuracy_pct = round((accuracy * 100), 4)
print(f"Predicted malicious packets with {accuracy_pct}% accuracy")
