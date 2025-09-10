# import pandas as pd

# def run_roughset(df, decision_column):
#     # Đây chỉ là minh họa rất cơ bản
#     attrs = [col for col in df.columns if col != decision_column]
#     decision_rules = []

#     for _, row in df.iterrows():
#         condition = " ∧ ".join([f"{attr}={row[attr]}" for attr in attrs])
#         decision = f"{decision_column}={row[decision_column]}"
#         decision_rules.append(f"Nếu {condition} thì {decision}")

#     return decision_rules

import pandas as pd
from itertools import combinations

def indiscernibility(df, attrs):
    """
    Tính indiscernibility relation cho tập thuộc tính attrs
    Trả về dictionary {key: tập dòng có cùng giá trị}
    """
    groups = df.groupby(attrs).groups
    return groups

def positive_region(df, condition_attrs, decision_attr):
    """
    Positive region của tập thuộc tính condition_attrs đối với decision_attr
    """
    ind_c = indiscernibility(df, condition_attrs)
    ind_d = indiscernibility(df, [decision_attr])
    pos = set()

    for _, c_indices in ind_c.items():
        for _, d_indices in ind_d.items():
            if set(c_indices).issubset(set(d_indices)):
                pos |= set(c_indices)
                break
    return pos

def find_reduct(df, decision_attr):
    """
    Tìm reduct nhỏ nhất (dùng brute force)
    """
    condition_attrs = [col for col in df.columns if col != decision_attr]
    full_pos = positive_region(df, condition_attrs, decision_attr)

    for r in range(1, len(condition_attrs) + 1):
        for subset in combinations(condition_attrs, r):
            pos = positive_region(df, list(subset), decision_attr)
            if pos == full_pos:
                return list(subset)  # trả về reduct đầu tiên tìm thấy
    return condition_attrs

def generate_rules(df, reduct, decision_attr):
    """
    Sinh luật quyết định từ reduct
    """
    rules = []
    grouped = df.groupby(reduct)

    for cond_values, group in grouped:
        decision_values = group[decision_attr].unique()
        condition = " ∧ ".join([f"{attr}={val}" for attr, val in zip(reduct, cond_values if isinstance(cond_values, tuple) else [cond_values])])
        for d in decision_values:
            rules.append(f"Nếu {condition} thì {decision_attr}={d}")
    return rules

def run_roughset(df, decision_attr):
    """
    Chạy Rough Set:
    - Tìm reduct
    - Sinh luật quyết định
    """
    reduct = find_reduct(df, decision_attr)
    rules = generate_rules(df, reduct, decision_attr)
    return reduct, rules
