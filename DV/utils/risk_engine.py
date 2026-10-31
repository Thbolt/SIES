def calculate_risk_score(data):

    score = 0

    # High transaction amount compared with normal amount
    avg = data.get("avg_transaction", 0)
    amount = data.get("amount", 0)

    if avg > 0:

        ratio = amount / avg

        if ratio >= 5:
            score += 25

        elif ratio >= 3:
            score += 18

        elif ratio >= 2:
            score += 10

    # New device
    if data.get("new_device") == "Yes":
        score += 12

    # New location
    if data.get("new_location") == "Yes":
        score += 12

    # New beneficiary
    if data.get("new_beneficiary") == "Yes":
        score += 15

    # IP change
    if data.get("ip_changed") == "Yes":
        score += 8

    # Unusual transaction time
    if data.get("unusual_time") == "Yes":
        score += 10

    # High transaction frequency
    if data.get("transaction_frequency") == "High":
        score += 10

    # Failed login attempts
    failed_logins = data.get("failed_logins", 0)

    if failed_logins >= 5:
        score += 18

    elif failed_logins >= 3:
        score += 12

    elif failed_logins >= 1:
        score += 5

    # No 2FA
    if data.get("two_factor") == "Disabled":
        score += 10

    # New accounts can receive slightly more risk
    account_age = data.get("account_age", 0)

    if account_age < 1:
        score += 5

    return min(score, 100)


def get_risk_level(score):

    if score <= 20:
        return "Very Low"

    elif score <= 40:
        return "Low"

    elif score <= 60:
        return "Moderate"

    elif score <= 80:
        return "High"

    else:
        return "Critical"


def get_risk_factors(data):

    factors = []

    amount = data.get("amount", 0)
    avg = data.get("avg_transaction", 0)

    if avg > 0 and amount >= avg * 2:
        factors.append(
            "Transaction amount is significantly higher than normal."
        )

    if data.get("new_device") == "Yes":
        factors.append(
            "Transaction initiated from a new device."
        )

    if data.get("new_location") == "Yes":
        factors.append(
            "Transaction originated from a new location."
        )

    if data.get("new_beneficiary") == "Yes":
        factors.append(
            "Money is being transferred to a newly added beneficiary."
        )

    if data.get("ip_changed") == "Yes":
        factors.append(
            "A change in IP address was detected."
        )

    if data.get("unusual_time") == "Yes":
        factors.append(
            "Transaction occurred at an unusual time."
        )

    if data.get("transaction_frequency") == "High":
        factors.append(
            "Unusually high transaction frequency was detected."
        )

    failed_logins = data.get("failed_logins", 0)

    if failed_logins > 0:
        factors.append(
            f"{failed_logins} failed login attempt(s) were recorded."
        )

    if data.get("two_factor") == "Disabled":
        factors.append(
            "Two-factor authentication is disabled."
        )

    return factors


def get_recommendations(score, data):

    recommendations = []

    if score >= 60:
        recommendations.append(
            "Verify the recent transaction before authorizing it."
        )

    if data.get("two_factor") == "Disabled":
        recommendations.append(
            "Enable two-factor authentication for the banking account."
        )

    if data.get("new_device") == "Yes":
        recommendations.append(
            "Verify that the new device belongs to the account owner."
        )

    if data.get("new_location") == "Yes":
        recommendations.append(
            "Confirm whether the new transaction location is legitimate."
        )

    if data.get("new_beneficiary") == "Yes":
        recommendations.append(
            "Verify the newly added beneficiary before transferring money."
        )

    if data.get("failed_logins", 0) >= 3:
        recommendations.append(
            "Change the banking password and review recent login activity."
        )

    if data.get("ip_changed") == "Yes":
        recommendations.append(
            "Review active sessions and sign out unknown devices."
        )

    if data.get("unusual_time") == "Yes":
        recommendations.append(
            "Review transactions performed outside normal usage hours."
        )

    if score >= 80:
        recommendations.append(
            "Contact the bank immediately if any transaction is unauthorized."
        )

    recommendations.append(
        "Enable instant transaction notifications."
    )

    recommendations.append(
        "Never share OTPs, PINs, CVVs or banking passwords."
    )

    return recommendations