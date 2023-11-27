import pandas as pd
import sklearn as sk
import sqlite3 as sql
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# load the data from excel into a pandas dataframe

# change this to LOCAL path to excel file 