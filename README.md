# Bank Fraud Risk Assessment Dashboard

A data visualization and fraud-risk assessment project developed using:

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn

## Project Objective

The system analyzes banking and transaction-related information to estimate
the likelihood of suspicious or fraudulent activity.

The dashboard provides:

- Risk score
- Risk classification
- Transaction analysis
- Risk factor analysis
- Possible threat assessment
- Security recommendations

## Project Structure

bank-fraud-dashboard/

    app.py

    data/
        paysim.csv

    model/
        fraud_model.pkl

    utils/
        preprocessing.py
        risk_engine.py

    requirements.txt

    README.md


## Installation

Open a terminal inside the project folder.

Create a virtual environment:

    python -m venv venv

Activate it on Windows:

    venv\Scripts\activate

Install requirements:

    pip install -r requirements.txt


## Run Dashboard

Run:

    streamlit run app.py


The application should automatically open in your browser.

Usually it will run at:

    http://localhost:8501


## Dataset

The project is designed to work with the PaySim fraud detection dataset.

Place the dataset inside:

    data/paysim.csv

If the dataset is not present, the application automatically generates sample
transaction data so that the dashboard can still run.


## Current Version

Version 1 uses a rule-based risk scoring system.

The score is calculated using:

- Transaction amount
- New device
- New location
- New beneficiary
- IP address change
- Unusual transaction time
- Failed login attempts
- Transaction frequency
- Two-factor authentication status


## Future Version

The next version will include a machine-learning model trained using the
PaySim dataset.

Candidate models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model evaluation will use:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC
- Precision-Recall analysis