# Heart Attack Risk Prediction using SVM

This repository contains a machine learning model that predicts the risk of a heart attack using Support Vector Machine (SVM). The project includes the dataset, the trained SVM model, and the Python code used to build and evaluate the model.

---

## Features
- Predicts the likelihood of a heart attack based on medical and lifestyle data.
- Uses a Support Vector Machine (SVM) for classification.
- Includes a Jupyter Notebook for data preprocessing, model training, and evaluation.
- Ready-to-use trained model.

---

## Dataset valuables 
The dataset contains medical and demographic information, including:
- **Age**: Age of the person
- **Gender**: Gender of the person
- **cp**: Chest Pain type
- **trtbps**: Resting blood pressure (in mm Hg)
- **chol**: Cholesterol in mg/dl fetched via BMI sensor
- **fbs**: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalachh**: Maximum heart rate achieved
- **exng**: Exercise-induced angina (1 = yes; 0 = no)
- **oldpeak**: Previous peak

The dataset is preprocessed to handle missing values, normalize numerical data, and encode categorical features.

---

## Files in the Repository
- `data/heart_attack_data.csv`: The dataset used for training and testing the model.
- `notebooks/Heart_Attack_Prediction.ipynb`: Jupyter Notebook containing the full workflow for preprocessing, training, and evaluation.
- `model/svm_heart_attack_model.pkl`: Pre-trained SVM model.
- `README.md`: This file providing an overview of the project.
- `heart_attack_model1.pkl`: The trained SVM model saved using Joblib.

---

## Requirements
To run the code and train the model, ensure you have the following dependencies installed:

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook (optional for running the notebook)

You can install all dependencies using the following command:
```bash
pip install -r requirements.txt
```

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your_username/svm-heart-attack-prediction.git
```

2. Navigate to the project directory:
```bash
cd svm-heart-attack-prediction
```

3. Open the Jupyter Notebook to explore and execute the code:
```bash
jupyter notebook notebooks/Heart_Attack_Prediction.ipynb
```

4. Alternatively, use the pre-trained model for predictions:
```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_attack_model1.pkl')

# Example data
example_data = pd.DataFrame({
    'age': [45],
    'gender': [1],
    'trtbps': [120],
    'chol': [220],
    'thalachh': [150],
    'exng': [0],
    'cp': [2],
    'fbs': [0],
    'restecg': [1],
    'oldpeak': [1.5]
})

# Make a prediction
prediction = model.predict(example_data)
print("Heart Attack Risk:", "High" if prediction[0] == 1 else "Low")
```

---

## Model Training
### Code Overview:
```python
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

# Save the model using joblib
joblib.dump(best_model, 'heart_attack_model1.pkl')

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
```
---

## Results
The model achieved the following performance on the test set:
- **Accuracy**: 86.5%
- **Precision**: 87%
- **Recall**: 85%
- **F1-Score**: 86%

---

## Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request with your improvements.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the creators of the dataset and the open-source community for providing the tools and inspiration to build this project.


      
