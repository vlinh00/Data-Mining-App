# # import pandas as pd
# # from itertools import combinations

# # def indiscernibility(df, attrs):
# #     """
# #     Tính indiscernibility relation cho tập thuộc tính attrs
# #     Trả về dictionary {key: tập dòng có cùng giá trị}
# #     """
# #     groups = df.groupby(attrs).groups
# #     return groups

# # def positive_region(df, condition_attrs, decision_attr):
# #     """
# #     Positive region của tập thuộc tính condition_attrs đối với decision_attr
# #     """
# #     ind_c = indiscernibility(df, condition_attrs)
# #     ind_d = indiscernibility(df, [decision_attr])
# #     pos = set()

# #     for _, c_indices in ind_c.items():
# #         for _, d_indices in ind_d.items():
# #             if set(c_indices).issubset(set(d_indices)):
# #                 pos |= set(c_indices)
# #                 break
# #     return pos

# # def find_reduct(df, decision_attr):
# #     """
# #     Tìm reduct nhỏ nhất (dùng brute force)
# #     """
# #     condition_attrs = [col for col in df.columns if col != decision_attr]
# #     full_pos = positive_region(df, condition_attrs, decision_attr)

# #     for r in range(1, len(condition_attrs) + 1):
# #         for subset in combinations(condition_attrs, r):
# #             pos = positive_region(df, list(subset), decision_attr)
# #             if pos == full_pos:
# #                 return list(subset)  # trả về reduct đầu tiên tìm thấy
# #     return condition_attrs

# # def generate_rules(df, reduct, decision_attr):
# #     """
# #     Sinh luật quyết định từ reduct
# #     """
# #     rules = []
# #     grouped = df.groupby(reduct)

# #     for cond_values, group in grouped:
# #         decision_values = group[decision_attr].unique()
# #         condition = " ∧ ".join([f"{attr}={val}" for attr, val in zip(reduct, cond_values if isinstance(cond_values, tuple) else [cond_values])])
# #         for d in decision_values:
# #             rules.append(f"Nếu {condition} thì {decision_attr}={d}")
# #     return rules

# # def run_roughset(df, decision_attr):
# #     """
# #     Chạy Rough Set:
# #     - Tìm reduct
# #     - Sinh luật quyết định
# #     """
# #     reduct = find_reduct(df, decision_attr)
# #     rules = generate_rules(df, reduct, decision_attr)
# #     return reduct, rules

# import pandas as pd
# from itertools import combinations

# def indiscernibility(df, attrs):
#     return df.groupby(attrs).groups

# def positive_region(df, condition_attrs, decision_attr):
#     ind_c = indiscernibility(df, condition_attrs)
#     ind_d = indiscernibility(df, [decision_attr])
#     pos = set()
#     for _, c_indices in ind_c.items():
#         for _, d_indices in ind_d.items():
#             if set(c_indices).issubset(set(d_indices)):
#                 pos |= set(c_indices)
#                 break
#     return pos

# def dependency_degree(df, condition_attrs, decision_attr):
#     pos = positive_region(df, condition_attrs, decision_attr)
#     return len(pos) / len(df)

# def lower_approx(df, X, attrs):
#     ind = indiscernibility(df, attrs)
#     lower = set()
#     for _, indices in ind.items():
#         if set(indices).issubset(X):
#             lower |= set(indices)
#     return lower

# def upper_approx(df, X, attrs):
#     ind = indiscernibility(df, attrs)
#     upper = set()
#     for _, indices in ind.items():
#         if set(indices).intersection(X):
#             upper |= set(indices)
#     return upper

# def find_reducts(df, decision_attr):
#     condition_attrs = [col for col in df.columns if col != decision_attr]
#     full_pos = positive_region(df, condition_attrs, decision_attr)
#     reducts = []
#     for r in range(1, len(condition_attrs) + 1):
#         for subset in combinations(condition_attrs, r):
#             pos = positive_region(df, list(subset), decision_attr)
#             if pos == full_pos:
#                 reducts.append(list(subset))
#     return reducts

# def generate_rules(df, reduct, decision_attr):
#     rules = []
#     grouped = df.groupby(reduct)
#     for cond_values, group in grouped:
#         decision_values = group[decision_attr].unique()
#         condition = " ∧ ".join([
#             f"{attr}={val}"
#             for attr, val in zip(reduct, cond_values if isinstance(cond_values, tuple) else [cond_values])
#         ])
#         for d in decision_values:
#             rules.append(f"Nếu {condition} thì {decision_attr}={d}")
#     return rules

# def run_roughset(df, decision_attr):
#     reducts = find_reducts(df, decision_attr)
#     all_rules = []
#     for reduct in reducts:
#         rules = generate_rules(df, reduct, decision_attr)
#         all_rules.append((reduct, rules))
#     return all_rules

import pandas as pd
from itertools import combinations

# =========================
# Indiscernibility relation
# =========================
def indiscernibility(df, attrs):
    return df.groupby(attrs).groups

# =========================
# Approximations
# =========================
def lower_approx(df, X, attrs):
    ind = indiscernibility(df, attrs)
    lower = set()
    for _, indices in ind.items():
        if set(indices).issubset(X):
            lower |= set(indices)
    return lower

def upper_approx(df, X, attrs):
    ind = indiscernibility(df, attrs)
    upper = set()
    for _, indices in ind.items():
        if set(indices).intersection(X):
            upper |= set(indices)
    return upper

# =========================
# Positive Region & Dependency
# =========================
def positive_region(df, condition_attrs, decision_attr):
    ind_c = indiscernibility(df, condition_attrs)
    ind_d = indiscernibility(df, [decision_attr])
    pos = set()
    for _, c_indices in ind_c.items():
        for _, d_indices in ind_d.items():
            if set(c_indices).issubset(set(d_indices)):
                pos |= set(c_indices)
                break
    return pos

def dependency_degree(df, condition_attrs, decision_attr):
    pos = positive_region(df, condition_attrs, decision_attr)
    return len(pos) / len(df)

# =========================
# Discernibility Matrix
# =========================
def discernibility_matrix(df, condition_attrs, decision_attr, as_dataframe=False):
    n = len(df)
    matrix = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if df.iloc[i][decision_attr] != df.iloc[j][decision_attr]:
                diff_attrs = {
                    a for a in condition_attrs if df.iloc[i][a] != df.iloc[j][a]
                }
                matrix[i][j] = diff_attrs
                matrix[j][i] = diff_attrs

    if as_dataframe:
        df_matrix = pd.DataFrame(matrix, columns=[f"x{j}" for j in range(n)])
        df_matrix.index = [f"x{i}" for i in range(n)]
        df_matrix = df_matrix.applymap(lambda x: ", ".join(x) if x else "∅")
        return df_matrix
    return matrix

# =========================
# Reducts
# =========================
def find_reducts(df, decision_attr, condition_attrs):
    full_pos = positive_region(df, condition_attrs, decision_attr)
    reducts = []
    for r in range(1, len(condition_attrs) + 1):
        for subset in combinations(condition_attrs, r):
            pos = positive_region(df, list(subset), decision_attr)
            if pos == full_pos:
                reducts.append(list(subset))
    return reducts

# =========================
# Rules
# =========================
def generate_rules(df, reduct, decision_attr):
    rules = []
    grouped = df.groupby(reduct)
    for cond_values, group in grouped:
        decision_values = group[decision_attr].unique()
        condition = " ∧ ".join(
            f"{attr}={val}"
            for attr, val in zip(reduct, cond_values if isinstance(cond_values, tuple) else [cond_values])
        )
        for d in decision_values:
            rules.append(f"Nếu {condition} thì {decision_attr}={d}")
    return rules

# =========================
# Run Rough Set
# =========================
def run_roughset(
    df,
    decision_attr,
    condition_attrs,
    show_matrix=False,
    max_reducts=None,
    approx_set=None,
):
    explanation = "=== Rough Set Analysis ===\n"

    # Step 1: Indiscernibility
    explanation += "\n[1] Quan hệ bất khả phân biệt (Indiscernibility classes):\n"
    ind = indiscernibility(df, condition_attrs)
    for key, idx in ind.items():
        explanation += f"  {key}: {list(idx)}\n"

    # Step 2: Approximations
    if approx_set:
        explanation += f"\n[2] Xấp xỉ tập X={approx_set} theo {condition_attrs}:\n"
        lower = lower_approx(df, approx_set, condition_attrs)
        upper = upper_approx(df, approx_set, condition_attrs)
        explanation += f"  Lower approx: {sorted(lower)}\n"
        explanation += f"  Upper approx: {sorted(upper)}\n"

    # Step 3: Positive region & Dependency
    explanation += "\n[3] Vùng dương & Độ phụ thuộc:\n"
    pos = positive_region(df, condition_attrs, decision_attr)
    gamma = dependency_degree(df, condition_attrs, decision_attr)
    explanation += f"  Positive region (POS): {sorted(pos)}\n"
    explanation += f"  Độ phụ thuộc γ = {gamma:.2f}\n"

    # Step 4: Discernibility Matrix
    df_matrix = None
    if show_matrix:
        explanation += "\n[4] Ma trận phân biệt:\n"
        df_matrix = discernibility_matrix(df, condition_attrs, decision_attr, as_dataframe=True)

    # Step 5: Reducts
    explanation += "\n[5] Reducts:\n"
    reducts = find_reducts(df, decision_attr, condition_attrs)
    if max_reducts:
        reducts = reducts[:max_reducts]
    for r in reducts:
        explanation += f"  {r}\n"

    # Step 6: Rules
    explanation += "\n[6] Luật từ reducts:\n"
    all_rules = []
    for reduct in reducts:
        rules = generate_rules(df, reduct, decision_attr)
        all_rules.append((reduct, rules))
        for r in rules:
            explanation += f"  {r}\n"

    return all_rules, explanation, df_matrix

