import pandas as pd
from functools import reduce

files = [
    ("submission_auto_best.csv", 2),
    ("submission_catboost_stack.csv", 1),
    ("submission_ensemble.csv", 3),
    ("submission_feature_engg.csv", 1),
    ("submission_metalearn.csv", 1),
    ("submission_survival_rules.csv", 3),
    ("submission_ensmbl_feature_engg.csv", 2),
]

dfs = []

for i, (fname, weight) in enumerate(files):
    df = pd.read_csv(fname)
    df = df.rename(columns={"Survived": f"pred_{i}"})
    dfs.append((df, weight))

merged = dfs[0][0]

for df, _ in dfs[1:]:
    merged = merged.merge(df, on="PassengerId")

merged["score"] = 0
total_weight = 0

for i, (_, weight) in enumerate(dfs):
    merged["score"] += weight * merged[f"pred_{i}"]
    total_weight += weight

merged["Survived"] = (merged["score"] >= (total_weight / 2)).astype(int)

final = merged[["PassengerId", "Survived"]]
final.to_csv("submission_blend_weighted.csv", index=False)

print("Saved submission_blend_weighted.csv")
