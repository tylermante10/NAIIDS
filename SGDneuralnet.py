# testing an SGD classifier
# author @tyler @mike

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from numpy import loadtxt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
import sqlite3 as sql
import numpy as np
import gc
# Training Section:

# create connection to training database. Adjust path as needed
# connection = sql.connect('train.db')
connection = sql.connect('/mnt/c/Users/mante/Downloads/train.db')

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
# print("Below is y array. 2D array of samples and features")
# print(features)
# print("Below is x array. 1D array of labels")
# print(labels)

# close connection and cursor
cursor.close()
connection.close()

# TESTING

# now reopen the connection and get features and labels for testing data
# create connection
# connection = sql.connect('test.db')
connection = sql.connect('/mnt/c/Users/mante/Downloads/test.db')
cursor = connection.cursor()

execute = f"SELECT {SQL_select} FROM test_1;"
cursor.execute(execute)

result_features = cursor.fetchall()

features_test = np.array(result_features, dtype='float32')


execute = f"SELECT Flag FROM test_1;"
cursor.execute(execute)

result_label = cursor.fetchall()


labels_test = np.array(result_label, dtype='float32')

labels_test = labels_test.flatten()

# learn machine
clf = SGDClassifier(shuffle = False) # adjusting shuffle parameter

# print(features_train[0,:])
# print(labels_train)


# Train the data
print("Training...")
clf.fit(features_train, labels_train.ravel()) 
print("Training success")

# Debug statements
# print(features_train.dtype)
# print(features_test.dtype)

print("\n")
# make predictions
labels_pred = clf.predict(features_test)

# make confusion matrix
print("Confusion matrix: ")
print(confusion_matrix(labels_test, labels_pred))

# classification report
cr = classification_report(labels_test, labels_pred)
print("Classification report: ")
print(cr)

clf_roc = RocCurveDisplay.from_estimator(clf, features_test, labels_test)
clf_roc.plot()
clf_roc.figure_.suptitle("ROC curve comparison")
# clf_roc.figure_.subplots_adjust(left=0.15, right=0.7)
clf_roc.figure_.set_size_inches(8, 6)
clf_roc.ax_.set_xlabel("False Positive Rate")
clf_roc.ax_.set_ylabel("True Positive Rate")
clf_roc.ax_.legend(loc="lower right")
clf_roc.figure_.savefig("ROC_curve_comparison.png")
print("ROC curve saved to ROC_curve_comparison.png")

# # DEBUG labels_pred

# Debug statements
# print(labels_pred.shape)
# print(labels_test.shape)

# # get accuracy
accuracy = clf.score(features_test, labels_test)
accuracy_pct = round((accuracy * 100), 4)
# # # get histogram of predictions
# print("Histogram of predictions: ")
# np.histogram(labels_pred)
# # View the histogram
# plt.show()

# false positive rate
# print("False positive rate: ", end="")
# run 10 epochs ??
# for i in range(10):
#     clf.partial_fit(features_train, labels_train, classes=np.unique(labels_train))

# # print the accuracy
print(f"Predicted malicious packets with {accuracy_pct}% accuracy")

# Assuming 'extracted_features' is the feature vector obtained from the SGD classifier

# Define a pyTorch neural net
# Convert NumPy arrays to PyTorch tensors
extracted_features_tensor = torch.tensor(labels_pred)
x_train_tensor = torch.tensor(labels_train)
# x_val_tensor = torch.tensor(x_val)
y_train_tensor = torch.tensor(features_train)
# y_val_tensor = torch.tensor(y_val)

# Create PyTorch DataLoader for training and validation sets
train_dataset = TensorDataset(extracted_features_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
# val_loader = DataLoader(val_dataset, batch_size=32)

# Define a neural network using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the model
input_size = labels_pred.shape[0]  # Adjust this according to the feature size
num_classes = 1  # number of output classes (just binary classifier)
model = NeuralNetwork(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())  # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss for each epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average validation loss and accuracy for each epoch
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_accuracy:.4f}")




