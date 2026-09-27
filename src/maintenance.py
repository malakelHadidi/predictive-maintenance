def classify_rul(rul):
    """
    Convert predicted RUL into a maintenance status.
    """

    if rul > 30:
        return "NORMAL"

    elif rul > 15:
        return "WARNING"

    else:
        return "CRITICAL"


def get_maintenance_action(status):
    """
    Return the recommended maintenance action
    associated with a maintenance status.
    """

    actions = {
        "NORMAL": "Continue operation and monitor condition.",
        "WARNING": "Plan maintenance and monitor the engine closely.",
        "CRITICAL": "Schedule maintenance as soon as possible."
    }

    return actions[status]


def get_maintenance_decision(rul):
    """
    Generate a complete maintenance decision
    from a predicted RUL value.
    """

    status = classify_rul(rul)

    action = get_maintenance_action(status)

    return {
        "predicted_RUL": float(rul),
        "status": status,
        "action": action
    }