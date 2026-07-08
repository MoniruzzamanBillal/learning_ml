import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score ,GridSearchCV
from sklearn.metrics import f1_score



np.random.seed(42)
n = 1500

df = pd.DataFrame({
    "likes":          np.random.randint(50, 20000, n),
    "comments":       np.random.randint(5, 2000, n),
    "shares":         np.random.randint(2, 1500, n),
    "hashtag_count":  np.random.randint(0, 30, n),
    "caption_length": np.random.randint(10, 300, n),
    "hour_posted":    np.random.randint(0, 24, n),
    "follower_count": np.random.randint(100, 500000, n),
    "is_video":       np.random.randint(0, 2, n),
})
df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["follower_count"] + 1)
df["shares_per_like"] = df["shares"] / (df["likes"] + 1)
df["engagement"] = df["likes"] + df["comments"] * 2 + df["shares"] * 3
threshold = df["engagement"].quantile(0.80)
df["is_viral"] = (df["engagement"] >= threshold).astype(int)
df = df.drop("engagement", axis=1)

X = df.drop("is_viral", axis=1)
y = df["is_viral"]

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Viral posts: {y.sum()} ({y.mean()*100:.1f}%)")

# ✅ Cross-validation in sklearn — one function call

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# cv=5 means 5-fold cross validation
# scoring='f1' means evaluate using F1 score each fold
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1')

print("=== 5-FOLD CROSS VALIDATION RESULTS ===")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score*100:.2f}%")

print(f"\n  Mean F1:  {cv_scores.mean()*100:.2f}%")
print(f"  Std Dev:  {cv_scores.std()*100:.2f}%")
print()
print("💡 Std Dev tells you how stable the model is.")
print("   Low std = consistent across different data slices = trustworthy.")
print("   High std = performance varies a lot = might be overfitting.")



# ✅ Compare single split vs cross-validation

# Single split (old way)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf.fit(X_train, y_train)
single_f1 = f1_score(y_test, rf.predict(X_test))

print("=== Single Split vs Cross-Validation ===")
print(f"  Single split F1:       {single_f1*100:.2f}%  ← could be lucky or unlucky")
print(f"  Cross-validation F1:   {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%  ← reliable")
print()
print("In real projects: always report cross-validation scores, not single split.")





# Define the parameter grid to search
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [5, 10, None],
    "min_samples_split": [2, 5],
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,              # 5-fold cross validation for each combination
    scoring='f1',
    n_jobs=-1,         # run in parallel
    verbose=1          # print progress
)

grid_search.fit(X_train, y_train)

print("\n=== GRID SEARCH RESULTS ===")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1:      {grid_search.best_score_*100:.2f}%")


# ✅ Use the best model to predict on test data

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_test_f1 = f1_score(y_test, y_pred_best)

print(f"Best model test F1:    {best_test_f1*100:.2f}%")
print(f"Default model test F1: {single_f1*100:.2f}%")
print(f"Improvement:           +{(best_test_f1 - single_f1)*100:.2f}%")

# ✅ See all results — not just the best

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[[
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_split",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]].sort_values("rank_test_score")

print("Top 5 parameter combinations:")
print(results_df.head(5).to_string(index=False))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ✅ Build a pipeline: Scale → Logistic Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(random_state=42, max_iter=1000))
])

# fit() automatically: fits scaler on X_train, then fits model on scaled X_train
lr_pipeline.fit(X_train, y_train)

# predict() automatically: scales X_test, then predicts
lr_pred = lr_pipeline.predict(X_test)

print("=== Logistic Regression Pipeline ===")
print(f"F1 Score: {f1_score(y_test, lr_pred)*100:.2f}%")
print()
print(classification_report(y_test, lr_pred, target_names=["Not Viral", "Viral"]))



# ✅ Random Forest pipeline (no scaler needed, but Pipeline still keeps things clean)

rf_pipeline = Pipeline([
    ('model', RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print(f"Random Forest Pipeline F1: {f1_score(y_test, rf_pred)*100:.2f}%")



# ✅ POWER MOVE: GridSearchCV + Pipeline together
# This is the real production pattern

# Note: when using a pipeline, prefix param names with the step name + '__'
# 'model__n_estimators' = the n_estimators param of the 'model' step

pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_pipeline = {
    'model__n_estimators': [50, 100],
    'model__max_depth':    [5, 10, None],
}

grid_pipeline = GridSearchCV(
    pipeline,
    param_grid_pipeline,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_pipeline.fit(X_train, y_train)

print(f"\nBest params:  {grid_pipeline.best_params_}")
print(f"Best CV F1:   {grid_pipeline.best_score_*100:.2f}%")
print(f"Test F1:      {f1_score(y_test, grid_pipeline.predict(X_test))*100:.2f}%")



# ✅ This is what a clean, production ML script looks like
# Every step is explicit, reproducible, and safe

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np

# --- 1. Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Define candidate models as pipelines ---
candidates = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(random_state=42, max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ))
    ]),
}

# --- 3. Evaluate each with cross-validation ---
print("=" * 55)
print("   MODEL COMPARISON — 5-Fold Cross Validation")
print("=" * 55)

best_name, best_score, best_pipeline = None, 0, None

for name, pipeline in candidates.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    mean, std = scores.mean(), scores.std()
    print(f"  {name:<25} CV F1: {mean*100:.2f}% ± {std*100:.2f}%")
    if mean > best_score:
        best_score = mean
        best_name  = name
        best_pipeline = pipeline

# --- 4. Train best model on full training data ---
print(f"\n🏆 Best model: {best_name}")
best_pipeline.fit(X_train, y_train)

# --- 5. Final evaluation on held-out test set ---
y_pred = best_pipeline.predict(X_test)
print(f"\nFinal Test F1: {f1_score(y_test, y_pred)*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=["Not Viral", "Viral"]))


# ✅ Predict on new data with the final pipeline — one clean call

new_posts = pd.DataFrame({
    "likes":           [12000, 300,  5000],
    "comments":        [890,   15,   420],
    "shares":          [650,   8,    180],
    "hashtag_count":   [5,     25,   10],
    "caption_length":  [120,   8,    200],
    "hour_posted":     [18,    3,    12],
    "follower_count":  [50000, 800,  12000],
    "is_video":        [1,     0,    1],
    "engagement_rate": [0.256, 0.41, 0.188],
    "shares_per_like": [0.054, 0.026, 0.036],
})

predictions   = best_pipeline.predict(new_posts)
probabilities = best_pipeline.predict_proba(new_posts)

print("=== FINAL VIRAL PREDICTOR ===")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "🔥 VIRAL" if pred == 1 else "❌ NOT VIRAL"
    print(f"Post {i+1}: {label}  ({prob[1]*100:.1f}% confidence)")
