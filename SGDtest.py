import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load your dataset
data = pd.read_excel('./LabelledData_2/3.xlsx')

# Step 2: Split your data
X = data.drop('Label', axis=1)  # Features
y = data['Flag']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("x", X)
print("y", y)
print("x_train", X_train)
print("y_train", y_train)
print("x_test", X_test)
print("y_test", y_test)

# Step 3: Create and train the SGDClassifier
sgd_clf = SGDClassifier()
# sgd_clf.fit(X_train, y_train)

print("sgd_clf", sgd_clf)

# # Step 4: Make predictions
# y_pred = sgd_clf.predict(X_test)

# # Step 5: Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# print(classification_report(y_test, y_pred))

# Step 6: Iterate and refine (if needed)
