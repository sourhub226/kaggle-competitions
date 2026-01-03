import pandas as pd
from functools import reduce

files = [
    "submission_auto_best.csv",
    "submission_catboost_stack.csv",
    "submission_ensemble.csv",
    "submission_feature_engg.csv",
    "submission_metalearn.csv",
    "submission_survival_rules.csv",
    "submission_ensmbl_feature_engg.csv",
]

dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    df = df.rename(columns={"Survived": f"pred_{i}"})
    dfs.append(df)

merged = reduce(lambda l, r: l.merge(r, on="PassengerId"), dfs)

pred_cols = [c for c in merged.columns if c.startswith("pred_")]

merged["Survived"] = (merged[pred_cols].sum(axis=1) >= (len(pred_cols) / 2)).astype(int)

merged[["PassengerId", "Survived"]].to_csv("submission_blend_hard.csv", index=False)