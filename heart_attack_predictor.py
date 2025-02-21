# training-a-svm-model-on-heart-attack-data-set-
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from IPython.display import display, HTML
import ipywidgets as widgets
#i upload my dataset o  google drive and then give the access to the code but you can download the data set on your computer and then give access directly by this code
#with open('/content/drive/MyDrive/Colab Notebooks/your_file.py', 'r') as f:
 #file_content = f.read()
#accessing the google drive
from google.colab import drive
drive.mount ('/content/drive')
#locating the dataset file
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/my doc/heart orginal.csv')
df.head()
# Load the dataset (assuming it's in a CSV file named 'heart_attack_risk.csv')
data = pd.read_csv('/content/drive/MyDrive/my doc/heart orginal.csv')
# Separate features and target variable
X = data.drop('output', axis=1)
y = data['output']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])
# Define hyperparameters for grid search
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
}
# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)
# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

 # Print the classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

 # Function to predict heart attack risk for new data
def predict_heart_attack_risk(new_data):
    return best_model.predict(new_data)
# Get the best model
best_model = grid_search.best_estimator_

# Save the model using joblib
joblib.dump(best_model, 'heart_attack_model1.pkl')  # Save with .pkl extension
# Load the trained model
model = joblib.load('heart_attack_model1.pkl') # load the model
# Load the trained model (make sure you've run the training code and saved the model first)
model = joblib.load('heart_attack_model1.pkl')

# Create input fields
fields = [
    "Age", "Sex (0:Female, 1:Male)", "Chest Pain Type (0-3)",
    "Resting Blood Pressure (mm Hg)", "Alcohol Drink (0:No, 1:Yes)",
    "Fasting Blood Sugar (0:No, 1:Yes)", "Resting ECG Results (0-2)",
    "Max Heart Rate", "Exercise Induced Angina (0:No, 1:Yes)",
    "Previous Peak", "Slope (0-2)", "Number of Major Vessels (0-3)",
    "Thalassemia (0-3)"
]

# Create input widgets
inputs = {field: widgets.FloatText(description=field) for field in fields}

# Create a button widget
button = widgets.Button(description="Predict")
output = widgets.Output()

# Display widgets
form = widgets.VBox(list(inputs.values()) + [button, output])
display(form)

# Function to make prediction
def predict(b):
    with output:
        output.clear_output()
        try:
            input_data = np.array([[inputs[field].value for field in fields]])
            prediction = model.predict(input_data)
            result = "High Risk" if prediction[0] == 1 else "Low Risk"
            display(HTML(f"<h3>Heart Attack Risk: {result}</h3>"))
        except ValueError:
            display(HTML("<h3 style='color: red;'>Error: Please enter valid numeric values for all fields.</h3>"))

# Attach the function to the button
button.on_click(predict)
