import pandas as pd

files = [
    ("submission_auto_best.csv", 1),
    ("submission_catboost_stack.csv", 1),
    ("submission_ensemble.csv", 1),
    ("submission_feature_engg.csv", 1),
    ("submission_metalearn.csv", 1),
    ("submission_survival_rules.csv", 1),
    ("submission_ensmbl_feature_engg.csv", 1),
]

base = pd.read_csv(files[0][0])[["PassengerId"]]
base["prob"] = 0
total_weight = 0

for fname, weight in files:
    df = pd.read_csv(fname)
    base["prob"] += weight * df["Survived"]
    total_weight += weight

base["prob"] /= total_weight
base["Survived"] = (base["prob"] >= 0.5).astype(int)

base[["PassengerId", "Survived"]].to_csv("submission_blend_soft.csv", index=False)
