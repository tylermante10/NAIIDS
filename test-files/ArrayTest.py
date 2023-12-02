# this tests for getting an array from a SQL database author @mike
# import all that good shit
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sqlite3 as sql
import numpy as np


# create connection
connection = sql.connect('LabelledData_2_1_2.db')

# create cursor
cursor = connection.cursor()


# LabelledData_2.db is the databse file
# the table is called combined_2

# this array is a 2D array, on sprint planning this is 3.d.i
# we want every column besides flag and label.
# List of columns to include
included_columns = ['Source_IP', 'Source_Port', 'Protocol', 'State', 'Row_dur', 'src_bytes',
                    'dest_bytes', 'Src_ttl', 'Dest_ttl', 'Src_loss', 'Dest_loss', 'Service', 'Src_bps', 'Dest_bps',
                    'Src_pkt_ct', 'Dst_pkt_ct', 'Src_win_val', 'Dst_win_val', 'Src_tcp_bsn', 'Dst_tcp_bsn',
                    'Src_mean_pkt_size', 'Dst_mean_pkt_size', 'Trans_depth', 'Resp_bdy_len', 'Src_jitter', 'Dest_jitter',
                    '"Start Time"', '"Last Time"', 'Src_arr_time', 'Dest_arr_time', 'Tcp_rtt', 'Synack_time', 'Ack_time',
                    'Is_sm_ips_ports', 'Ct_state_ttl', 'Ct_http_f', 'login_ftp', 'Ct_ftp_cmd', 'Ct_srv_src', 'Ct_srv_dest',
                    'Ct_dest_ltm', 'Ct_str_ltm', 'Ct_src_dport_ltm', 'Ct_dest_sport_ltm', 'Ct_dest_src_ltm', 'Dest_IP', 'Dest_Port']

# Combine column names into a single string
SQL_select = ', '.join(included_columns)

# seems to be some edge case where it gets rows that are filled with None? so have it not include those
execute = f"SELECT {SQL_select} FROM data WHERE Source_IP <> 'None';"
cursor.execute(execute)

# get the result of that execute. this will be the 2D array with every column except flag
result_y = cursor.fetchall()

# let's repeat but for a one dimensional array
# we want the flag column (or label, we can easily switch it out)
# create execute string
execute = f"SELECT label FROM data WHERE Source_IP <> 'None';"
cursor.execute(execute)

result_x = cursor.fetchall()




# we can convert the result to a array
# this array is a 2D array if sample number and feature, on sprint planning this is 3.d.i
result_y_array = np.array(result_y)

# this array is a 1d array of labels, on sprint planning this is 3.d.ii
result_x_array = np.array(result_x)

print("Below is y array. 2D array of samples and features")
print(result_y_array)
print("Below is x array. 1D array of labels")
print(result_x_array)




# close connection and cursor
cursor.close()
connection.close()
