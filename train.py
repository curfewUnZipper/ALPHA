import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBaseline
from surprise.model_selection import cross_validate
import joblib

# =============================
# Load your dataset
# =============================
df = pd.read_csv("data.csv")

# Convert to long format for Surprise
melt_df = df.melt(id_vars=["Job_Role"], var_name="Feature", value_name="Score")

# =============================
# Prepare Surprise dataset
# =============================
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(melt_df[["Job_Role", "Feature", "Score"]], reader)
trainset = data.build_full_trainset()

# =============================
# Model selection
# =============================

# KNNBaseline gives stronger feature-weight correlation than SVD
sim_options = {
    'name': 'pearson_baseline',  # centers data before computing similarity
    'user_based': True  # comparing careers (users)
}

model = KNNBaseline(sim_options=sim_options)

# Train the model
model.fit(trainset)

# Optionally evaluate
cross_validate(model, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# =============================
# Save trained model + metadata
# =============================
joblib.dump(model, "career_recommender_model.pkl")
joblib.dump(df["Job_Role"].tolist(), "job_roles.pkl")
joblib.dump(melt_df["Feature"].unique().tolist(), "features.pkl")

print("✅ Model training complete and saved successfully.")
