import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBaseline 
from surprise import Dataset, Reader, SVD, KNNBasic, NMF, SlopeOne, CoClustering, BaselineOnly
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, f1_score
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
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# =============================
# Model selection
# =============================

# KNNBaseline gives stronger feature-weight correlation than SVD
sim_options = {
    'name': 'pearson_baseline',  # centers data before computing similarity
    'user_based': True  # comparing careers (users)
}

model = KNNBaseline(sim_options=sim_options)
model = KNNBaseline(sim_options=sim_options)

# Train the model
model.fit(trainset)

# # Default: SVD for matrix factorization
# model = SVD()
# model.fit(trainset)

# sim_options = {
#     'name': 'cosine',      # or 'pearson'
#     'user_based': True     # True = compare users; False = compare jobs
# }
# model = KNNBasic(sim_options=sim_options)
# model.fit(trainset)
# predictions = model.test(testset)


# model = NMF()
# model.fit(trainset)
# predictions = model.test(testset)

# from surprise import SlopeOne

# model = SlopeOne()
# model.fit(trainset)
# predictions = model.test(testset)



# from surprise import CoClustering

# model = CoClustering()
# model.fit(trainset)
# predictions = model.test(testset)


# from surprise import BaselineOnly

# model = BaselineOnly()
# model.fit(trainset)
# predictions = model.test(testset)



# =============================
# Evaluate model (RMSE, MAE)
# =============================
results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# =============================
# Additional classification-style metrics
# =============================
predictions = model.test(testset)

# Convert rating predictions to binary (>=3 considered "positive")
y_true = [1 if pred.r_ui >= 3 else 0 for pred in predictions]
y_pred = [1 if pred.est >= 3 else 0 for pred in predictions]

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n🔍 Additional Evaluation Metrics")
print(f"Accuracy: {acc:.3f}")
print(f"Recall:   {rec:.3f}")
print(f"F1-score: {f1:.3f}")

# =============================
# Save trained model + metadata
# =============================
joblib.dump(model, "career_recommender_model.pkl")
joblib.dump(df["Job_Role"].tolist(), "job_roles.pkl")
joblib.dump(melt_df["Feature"].unique().tolist(), "features.pkl")

print("\n✅ Model training complete and saved successfully.")
