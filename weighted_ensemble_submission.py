import pandas as pd

file1_path = "cnn-transformer.csv"
file2_path = "single_frame.csv"

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

weighted_risk = 0.65 * df1["risk"] + 0.35 * df2["risk"]

new_submission = df1.copy()
new_submission["risk"] = weighted_risk

output_path = "weighted_ensemble_submission.csv"
new_submission.to_csv(output_path, index=False)
