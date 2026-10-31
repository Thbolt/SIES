import pandas as pd
import numpy as np


def load_transaction_data(file_path):

    try:

        df = pd.read_csv(file_path)

        # PaySim dataset format
        if "type" in df.columns and "amount" in df.columns:

            # Take a smaller sample for dashboard speed
            if len(df) > 1000:
                df = df.sample(
                    n=1000,
                    random_state=42
                )

            df = df.reset_index(drop=True)

            df["transaction_id"] = (
                df.index + 1
            )

            return df

        else:

            return create_sample_data()

    except Exception as error:

        print(
            "Error loading dataset:",
            error
        )

        return create_sample_data()


def create_sample_data():

    np.random.seed(42)

    transaction_types = [
        "UPI",
        "Card",
        "Bank Transfer",
        "Cash Withdrawal",
        "Online Payment"
    ]

    rows = []

    for i in range(1, 31):

        transaction_type = np.random.choice(
            transaction_types
        )

        amount = np.random.randint(
            500,
            15000
        )

        rows.append({
            "transaction_id": i,
            "type": transaction_type,
            "amount": amount
        })

    return pd.DataFrame(rows)


def preprocess_paysim(df):

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    available_columns = [
        column
        for column in required_columns
        if column in df.columns
    ]

    processed = df[
        available_columns
    ].copy()

    # Balance changes

    if (
        "oldbalanceOrg" in processed.columns
        and
        "newbalanceOrig" in processed.columns
    ):

        processed["origin_balance_change"] = (
            processed["oldbalanceOrg"]
            -
            processed["newbalanceOrig"]
        )

    if (
        "oldbalanceDest" in processed.columns
        and
        "newbalanceDest" in processed.columns
    ):

        processed["destination_balance_change"] = (
            processed["newbalanceDest"]
            -
            processed["oldbalanceDest"]
        )

    return processed