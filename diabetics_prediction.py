import numpy as np  # To make numpy arrays for processing
import pandas as pd  # Creating Data Frames
from sklearn.preprocessing import StandardScaler  # For Standardizing Data
from sklearn.model_selection import train_test_split  # Splitting Data into Training & Testing
from sklearn import svm  # SVM - Support Vector Machine
from sklearn.metrics import accuracy_score  # For Accuracy Calculation

# Load the Diabetes Dataset (Ensure 'diabetes.csv' is in the same folder)
diabetics_dataset = pd.read_csv('diabetes.csv')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(diabetics_dataset.head())

# Determine number of rows and columns
print("\nDataset Shape:", diabetics_dataset.shape)

# Statistical Summary of Data
print("\nStatistical Summary:")
print(diabetics_dataset.describe())

# Count of Outcome Labels (0 = Not Diabetic, 1 = Diabetic)
print("\nOutcome Counts:")
print(diabetics_dataset['Outcome'].value_counts())

# Mean Values Grouped by Outcome
print("\nMean values grouped by Outcome:")
print(diabetics_dataset.groupby('Outcome').mean())

# Separating Features (X) and Labels (Y)
X = diabetics_dataset.drop(columns='Outcome', axis=1)
Y = diabetics_dataset['Outcome']

print("\nFeature Data (X):")
print(X)

print("\nLabels (Y):")
print(Y)

# Standardizing the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_scaled)

# Splitting Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize Support Vector Machine Classifier
classifier = svm.SVC(kernel='linear')

# Train the SVM Classifier
classifier.fit(X_train, Y_train)

# Accuracy Score on Training Data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('\nAccuracy Score on Training Data:', training_data_accuracy)

# Accuracy Score on Test Data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score on Test Data:', testing_data_accuracy)

# Predicting for a New Input Data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array for model prediction (1 sample, multiple features)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the Input Data
std_data = scaler.transform(input_data_reshaped)
print("\nStandardized Input Data:")
print(std_data)

# Make Prediction
prediction = classifier.predict(std_data)

# Display Prediction Result
if prediction[0] == 0:
    print("\nThe Person is NOT Diabetic.")
else:
    print("\nThe Person is Diabetic.")
