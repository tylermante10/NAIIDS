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

# copy code from ArrayTest.py

# create connection
connection = sql.connect('train.db')

# create cursor
cursor = connection.cursor()


# LabelledData_2.db is the databse file
# the table is called data

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
execute = f"SELECT {SQL_select} FROM train_1 LIMIT 100000;"
cursor.execute(execute)


# get the result of that execute. this will be the 2D array with every column except flag
result_features = cursor.fetchall()

print("47")
# let's repeat but for a one dimensional array
# we want the flag column (or label, we can easily switch it out)
# create execute string
execute = f"SELECT Flag FROM train_1 LIMIT 100000;"
cursor.execute(execute)

result_label = cursor.fetchall()

print("56")
# we can convert the result to a array
# this array is a 2D array if sample number and feature, on sprint planning this is 3.d.i

gc.collect()


features_train = np.array(result_features)

# this array is a 1d array of labels, on sprint planning this is 3.d.ii
labels_train = np.array(result_label)


print("65")
# print("Below is y array. 2D array of samples and features")
# print(features)
# print("Below is x array. 1D array of labels")
# print(labels)

# close connection and cursor
cursor.close()
connection.close()


# now   let's get into actually messing with the data and SGD

# split the data into training and testing data
# test_size = 0.5, means 50/50 test and train
# we do NOT want to use random_state, or shuffle, because we want to keep the order of the data
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

# learn machine
clf = SGDClassifier() # running it raw with default parameters

# print(features_train[0,:])
# print(labels_train)


# Assuming features_train is your NumPy array

# # Check for NaN values in the NumPy array
# nan_mask = np.isnan(features_train.astype(float))

# # Sum the NaN values along each column
# nan_sum_per_column = np.sum(nan_mask, axis=0)

# # Print the count of NaN values for each column
# print("Number of NaN values per column:")
# print(nan_sum_per_column)

# train the data

clf.fit(features_train, labels_train.ravel()) # .ravel() is to get rid of a warning, it's not important

# make predictions
# y_pred = clf.predict(X_test)

# print(y_pred)

# # get accuracy
# accuracy = clf.score(y_test, y_pred)
# #print the accuracy
# print(accuracy)

# we should have trained data by now?
