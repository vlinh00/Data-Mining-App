import pandas as pd

def run_roughset(df, decision_column):
    # Đây chỉ là minh họa rất cơ bản
    attrs = [col for col in df.columns if col != decision_column]
    decision_rules = []

    for _, row in df.iterrows():
        condition = " ∧ ".join([f"{attr}={row[attr]}" for attr in attrs])
        decision = f"{decision_column}={row[decision_column]}"
        decision_rules.append(f"Nếu {condition} thì {decision}")

    return decision_rules
