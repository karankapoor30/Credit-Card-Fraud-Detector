# Credit-Card-Fraud-Detector
A fraud detection model using Random Forest Classification to identify fraudulent transactions. Handles imbalanced data, provides feature importance, and visualizes results with a confusion matrix &amp; ROC curve. Built with Python, Scikit-Learn, and Pandas.

📌 Overview
This project implements a Credit Card Fraud Detection model using Random Forest Classification. The goal is to identify fraudulent transactions based on historical transaction data while handling class imbalances and ensuring high accuracy.

🚀 Features
Random Forest Classifier for robust and efficient fraud detection.
Handles Imbalanced Data using undersampling/oversampling techniques.
Feature Importance Analysis for better interpretability.
Evaluation Metrics: Confusion Matrix, Precision-Recall.

📂 Dataset
The model is trained on the Kaggle Credit Card Fraud Detection dataset, containing real anonymized credit card transactions. It includes 284,807 transactions with only 492 fraud cases (~0.17% fraud rate), making it highly imbalanced.

Dataset Columns
Time: Transaction time in seconds.
V1 - V28: PCA-transformed features for anonymity.
Amount: Transaction amount.
Class: Target variable (0 = Legitimate, 1 = Fraudulent).

🛠 Technologies Used
Python
Scikit-Learn
Pandas, NumPy
Matplotlib, Seaborn
Jupyter Notebook

🔧 Installation
Clone this repository:
git clone https://github.com/karankapoor30/credit-card-fraud-detection.git

Navigate to the project folder:
cd Credit-Card-Fraud-Detector

Install dependencies:
pip install -r requirements.txt

Run the Jupyter Notebook:
jupyter notebook main.ipynb

📊 Data Preprocessing
Handling Missing Values: The dataset does not contain missing values.
Feature Scaling: Standardized Amount using MinMaxScaler.
Class Imbalance Handling:
Undersampling: Reduces non-fraud samples.
Oversampling: Uses SMOTE (Synthetic Minority Over-sampling Technique).

🤖 Model Training
Splitting Data: 80% Training, 20% Testing.
Classifier: Random Forest with optimized hyperparameters.
Hyperparameter Tuning:
Number of trees (n_estimators)
Max depth (max_depth)
Minimum samples split (min_samples_split)
Minimum samples per leaf (min_samples_leaf)

Training Execution:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

📈 Model Evaluation
Confusion Matrix: Measures true positives, false positives, etc.
Precision-Recall Curve: Evaluates effectiveness in detecting fraud.

📌 Results
Accuracy: 99.5%
Precision: 90.2%
Recall: 87.3%
F1-Score: 88.7%

📢 Future Improvements
Implement Deep Learning (LSTM, CNN) for better fraud detection.
Test with other classifiers like XGBoost and SVM.
Deploy as a real-time fraud detection API.

🤝 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.


📌 Feel free to reach out for suggestions or improvements! 🚀
