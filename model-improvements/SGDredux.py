# testing an SGD classifier
# author @tyler @mike

# import libraries
import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sqlite3 as sql
import numpy as np
import gc
## Training Section:

# create connection to training database. TODO Adjust path below:
connection = sql.connect('/mnt/c/Users/mante/Downloads/train.db')

# create cursor
cursor = connection.cursor()

# this array is a 2D array, on sprint planning this is 3.d.i
# Initially, we want every column besides flag and label.
included_columns = ['Src_ip_A', 'Src_ip_B', 'Src_ip_C', 'Src_ip_D', 'Source_Port', 'Protocol_feature',
                    'State_feature', 'Row_dur', 'src_bytes', 'dest_bytes', 'Src_ttl', 'Dest_ttl', 'Src_loss', 'Dest_loss',
                    'Service_feature', 'Src_bps', 'Dest_bps', 'Src_pkt_ct', 'Dst_pkt_ct', 'Src_win_val', 'Dst_win_val',
                    'Src_tcp_bsn', 'Dst_tcp_bsn', 'Src_mean_pkt_size', 'Dst_mean_pkt_size', 'Trans_depth', 'Resp_bdy_len',
                    'Src_jitter', 'Dest_jitter', '"Start Time"', '"Last Time"', 'Src_arr_time', 'Dest_arr_time', 'Tcp_rtt',
                    'Synack_time', 'Ack_time', 'Is_sm_ips_ports', 'Ct_state_ttl', 'Ct_http_f', 'login_ftp', 'Ct_ftp_cmd',
                    'Ct_srv_src', 'Ct_srv_dest', 'Ct_dest_ltm', 'Ct_str_ltm', 'Ct_src_dport_ltm', 'Ct_dest_sport_ltm',
                    'Ct_dest_src_ltm', 'Dest_ip_A', 'Dest_ip_B', 'Dest_ip_C', 'Dest_ip_D', 'Dest_Port']
# Modified included_columns as per feature extraction methods - TODO: Uncomment and adjust as needed with above
# included_columns = ['Src_ip_D', 'Source_Port', 'Protocol_feature',
#                     'State_feature', 'Row_dur', 'src_bytes', 'dest_bytes', 'Src_ttl', 'Dest_ttl', 'Src_loss', 'Dest_loss',
#                     'Src_bps', 'Dest_bps', 'Src_pkt_ct', 'Dst_pkt_ct', 
#                     'Src_mean_pkt_size', 'Dst_mean_pkt_size',
#                     'Src_jitter', 'Dest_jitter', '"Start Time"', '"Last Time"', 'Src_arr_time', 'Dest_arr_time', 'Tcp_rtt',
#                     'Synack_time', 'Ack_time', 'Ct_state_ttl',
#                     'Ct_srv_src', 'Ct_srv_dest', 'Ct_dest_ltm', 'Ct_str_ltm', 'Ct_src_dport_ltm', 'Ct_dest_sport_ltm',
#                     'Ct_dest_src_ltm', 'Dest_ip_A', 'Dest_ip_B', 'Dest_ip_C', 'Dest_ip_D', 'Dest_Port']

# Combine column names into a single string
SQL_select = ', '.join(included_columns)

# Querying the training database with columns for features
execute = f"SELECT {SQL_select} FROM train_1;"
cursor.execute(execute)

# get the result of that execute. this will be the 2D array with every column except flag
result_features = cursor.fetchall()
print("Features selection success")

# let's repeat but for a one dimensional array
# we want the flag column (or label, we can easily switch it out)
# create execute string
execute = f"SELECT Flag FROM train_1;"
cursor.execute(execute)

result_label = cursor.fetchall()
print("Training label selection success")

# Convert results to arrays
# this array is a 2D array if sample number and feature, on sprint planning this is 3.d.i
features_train = np.array(result_features, dtype='float32')

# this array is a 1d array of labels, on sprint planning this is 3.d.ii
labels_train = np.array(result_label, dtype='float32')
labels_train = labels_train.flatten()

print("Features and labels successfully converted to numpy arrays")

# close connection and cursor
cursor.close()
connection.close()

## TESTING Section

# now reopen the connection and get features and labels for testing data
# create connection to testing database. Adjust path below
# connection = sql.connect('test.db')
connection = sql.connect('/mnt/c/Users/mante/Downloads/test.db')
cursor = connection.cursor()

execute = f"SELECT {SQL_select} FROM test_1;"
cursor.execute(execute)

# get the result of that execute. this will be the 2D array with every column except flag
result_features = cursor.fetchall()
features_test = np.array(result_features, dtype='float32')

# this array is a 1d array of labels, on sprint planning this is 3.d.ii but for testing
execute = f"SELECT Flag FROM test_1;"
cursor.execute(execute)

# fetchall, cast to array
result_label = cursor.fetchall()
labels_test = np.array(result_label, dtype='float32')
labels_test = labels_test.flatten()

## ATTEMPT 1: Variance Thresholding

## Apply Variance Thresholding to select features - TODO: Comment lines 108-126 to run others
threshold = 0.3  # Define the threshold here
selector = VarianceThreshold(threshold=threshold)
selected_features = selector.fit_transform(features_train)
print(selected_features.shape)

# Creates a boolean mask of selected features
selected_mask = selector.get_support()

# Filter original column names based on the mask
selected_column_names = [col_name for idx, col_name in enumerate(included_columns) if selected_mask[idx]]

# Print out the selected column names
print("Selected features (column names):")
for col_name in selected_column_names:
    print(col_name)

# # Apply the selected features to your data
selected_features_train = selector.transform(features_train)
selected_features_test = selector.transform(features_test)

## ATTEMPT 2: Information Gain - uncomment to run - TODO: Uncomment lines 128-132 to run attempt 2 (comment lines 108-126)
# importances = mutual_info_classif(features_train, labels_train)
# feat_importances = pd.Series(importances, included_columns[0:len(included_columns)])
# feat_importances.plot(kind='barh', color='blue')
# plt.show()

## ATTEMPT 3: Exhaustive Feature Selection - TODO uncomment lines 134-EOF to see logic (WARNING: Do not run- takes too long)
# print("Exhaustive Feature Selection commencing...")
# efs = EFS(SGDClassifier(shuffle = False, max_iter = 1000000000), min_features = 1, max_features = 53, scoring = 'roc_auc', print_progress = True, cv = 2)
# efs.fit(features_train, labels_train.ravel())

# print('Best roc_auc: %.2f' % efs.best_score_)
# print('Best subset (indices):', efs.best_idx_)
# print('Best subset (corresponding names):', efs.best_feature_names_)
# print('Mask for features selected:', efs.best_mask_)
# print('Total number of features:', efs.n_features_in_)
# print('Total number of subsets:', efs.total_features_)
# print('Total number of features in subsets:', efs.subset_dim_)
# print('Total number of subsets that were checked:', efs.total_feature_names_)
# print('Total number of features that were checked:', efs.total_feature_names_)
# print('Total time taken to perform exhaustive search:', efs.total_time_)
# print('Total time taken to perform exhaustive search (formatted):', efs.total_time_formatted_)
# print('Total time taken to perform exhaustive search (seconds):', efs.total_time_secs_)
# print('Total time taken to perform exhaustive search (minutes):', efs.total_time_mins_)
# print('Total time taken to perform exhaustive search (hours):', efs.total_time_hours_)
# print('Total time taken to perform exhaustive search (days):', efs.total_time_days_) # This is why we couldn't complete 