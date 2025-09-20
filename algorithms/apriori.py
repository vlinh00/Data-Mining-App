# import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules
# import matplotlib.pyplot as plt

# def run_apriori(df, min_support=0.5, min_confidence=0.7):
#     """
#     Chạy thuật toán Apriori.
#     Input: df (one-hot encoded DataFrame, chỉ chứa 0/1)
#     Output: rules (DataFrame), fig (biểu đồ support vs confidence)
#     """

#     # Ép kiểu về 0/1
#     df = df.astype(bool).astype(int)

#     # Tìm tập phổ biến
#     frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

#     # Sinh luật kết hợp
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

#     # Vẽ biểu đồ support - confidence
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
#     ax.set_xlabel("Support")
#     ax.set_ylabel("Confidence")
#     ax.set_title("Biểu đồ Support vs Confidence (màu = Lift)")
#     fig.colorbar(ax.collections[0], ax=ax, label="Lift")

#     return rules, fig

import re
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ====================
# HÀM CỐT LÕI
# ====================
def get_support(itemset, transactions):
    s_item = set(itemset)
    count = sum(1 for t in transactions if s_item.issubset(set(t)))
    return count / len(transactions) if transactions else 0.0

def format_item(itemset):
    return "{" + ",".join(itemset) + "}"

def apriori_with_explanations(transactions, minsup):
    all_items = sorted(set(i for t in transactions for i in t))
    level_L = {}
    frequent = []
    explanation_log = ""

    k = 1
    current_candidates = [[i] for i in all_items]

    while current_candidates:
        explanation_log += f"\n--- Các tập ứng cử viên có {k} phần tử (C{k}): ---\n"
        Lk = []

        for cand in current_candidates:
            sup = get_support(cand, transactions)
            status = "Phổ biến" if sup >= minsup else "Không phổ biến"
            explanation_log += f"{format_item(cand)} -> SP={sup:.2f} ; {status}\n"
            if sup >= minsup:
                Lk.append((cand, sup))
                frequent.append((cand, sup))

        if Lk:
            Lk_items = [cand for cand, _ in Lk]
            level_L[k] = Lk_items
            Lk_str = ", ".join(format_item(c) for c in Lk_items)
            explanation_log += f"=> Tập phổ biến có {k} phần tử (L{k}): {Lk_str}\n"
        else:
            explanation_log += f"=> Không có tập phổ biến có {k} phần tử (L{k} = ∅)\n"

        # Sinh C{k+1}
        next_candidates = []
        Lk_only = [cand for cand, _ in Lk]
        for i in range(len(Lk_only)):
            for j in range(i+1, len(Lk_only)):
                union_set = sorted(set(Lk_only[i]) | set(Lk_only[j]))
                if len(union_set) == k+1 and union_set not in next_candidates:
                    all_subsets_exist = all(
                        list(subset) in Lk_only for subset in combinations(union_set, k)
                    )
                    if all_subsets_exist:
                        next_candidates.append(union_set)

        current_candidates = next_candidates
        k += 1

    return frequent, level_L, explanation_log

def generate_rules_from_freq(frequent_itemsets, transactions, minconf):
    rules = []
    sup_map = {tuple(sorted(itemset)): sup for itemset, sup in frequent_itemsets}

    for itemset, sup in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                sup_ant = sup_map.get(antecedent)
                if sup_ant is None:
                    sup_ant = get_support(list(antecedent), transactions)
                conf = sup / sup_ant if sup_ant > 0 else 0

                # lift
                sup_cons = sup_map.get(consequent)
                if sup_cons is None:
                    sup_cons = get_support(list(consequent), transactions)
                lift = sup / (sup_ant * sup_cons) if sup_ant > 0 and sup_cons > 0 else None

                if conf >= minconf:
                    rules.append({
                        "antecedent": list(antecedent),
                        "consequent": list(consequent),
                        "support": round(sup, 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4) if lift else None
                    })
    return pd.DataFrame(rules)

# ====================
# VẼ ĐỒ THỊ
# ====================
def plot_scatter(rules_df):
    if rules_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(
        rules_df["support"], rules_df["confidence"],
        c=rules_df["lift"], cmap="viridis", alpha=0.7
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Biểu đồ Support–Confidence (màu theo Lift)")
    plt.colorbar(sc, label="Lift", ax=ax)
    return fig

def plot_top_rules(rules_df, top_n=10, metric="confidence"):
    if rules_df.empty:
        return None
    rules_sorted = rules_df.sort_values(metric, ascending=False).head(top_n)
    labels = [
        f"{','.join(rules_sorted.iloc[i]['antecedent'])} → {','.join(rules_sorted.iloc[i]['consequent'])}"
        for i in range(len(rules_sorted))
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, rules_sorted[metric], color="skyblue")
    ax.set_xlabel(metric.capitalize())
    ax.set_title(f"Top {top_n} luật theo {metric.capitalize()}")
    ax.invert_yaxis()
    return fig

# ====================
# API chính cho app.py
# ====================
def run_apriori(df: pd.DataFrame, min_support=0.5, min_confidence=0.7):
    # chuẩn hóa df -> transactions
    if df.shape[1] == 2:
        tid_col, item_col = df.columns[0], df.columns[1]
        grouped = df.groupby(tid_col)[item_col].apply(list).tolist()
        transactions = []
        for t in grouped:
            items = []
            for it in t:
                if pd.isna(it): continue
                s = str(it).strip()
                parts = [p for p in re.split(r"[,\s]+", s) if p]
                items.extend(parts)
            # bỏ trùng
            seen, out = set(), []
            for x in items:
                if x not in seen:
                    seen.add(x); out.append(x)
            transactions.append(out)
    else:
        cols = df.columns
        transactions = []
        for _, row in df.iterrows():
            items = [c for c, v in zip(cols, row.values) if v > 0]
            transactions.append(items)

    # chạy apriori
    frequent, _, log = apriori_with_explanations(transactions, min_support)
    rules_df = generate_rules_from_freq(frequent, transactions, min_confidence)

    # vẽ đồ thị
    fig1 = plot_scatter(rules_df)
    fig2 = plot_top_rules(rules_df, top_n=10, metric="confidence")

    return rules_df, fig1, fig2, log
