import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

# =============================
# Load your data
# =============================
df = pd.read_csv("data.csv")  # your CSV file

# Convert to long format for Surprise
melt_df = df.melt(id_vars=["Job_Role"], var_name="Feature", value_name="Score")

# =============================
# Train Surprise model
# =============================
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(melt_df[["Job_Role", "Feature", "Score"]], reader)
trainset = data.build_full_trainset()

model = SVD()  # Try others: KNNBasic, NMF, SlopeOne, CoClustering, BaselineOnly
model.fit(trainset)

# =============================
# Save trained model and feature list
# =============================
joblib.dump(model, "career_recommender_model.pkl")
joblib.dump(df["Job_Role"].tolist(), "job_roles.pkl")
joblib.dump(melt_df["Feature"].unique().tolist(), "features.pkl")

print("✅ Model training complete and saved successfully.")
