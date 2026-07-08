import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier





np.random.seed(42)
n = 1500

df = pd.DataFrame({
    "likes":           np.random.randint(50, 20000, n),
    "comments":        np.random.randint(5, 2000, n),
    "shares":          np.random.randint(2, 1500, n),
    "hashtag_count":   np.random.randint(0, 30, n),
    "caption_length":  np.random.randint(10, 300, n),
    "hour_posted":     np.random.randint(0, 24, n),
    "follower_count":  np.random.randint(100, 500000, n),
    "is_video":        np.random.randint(0, 2, n),       # 0=image, 1=video
})



df["engagement"] = df["likes"] + df["comments"] * 2 + df["shares"] * 3
threshold = df["engagement"].quantile(0.80)
df["is_viral"] = (df["engagement"] >= threshold).astype(int)
df = df.drop("engagement", axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Viral posts: {df['is_viral'].sum()} ({df['is_viral'].mean()*100:.1f}%)")
df.head()


X = df.drop("is_viral", axis=1)
y = df["is_viral"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)




train_f1_full = f1_score(y_train, tree_full.predict(X_train))
test_f1_full  = f1_score(y_test,  tree_full.predict(X_test))


print("=== FULLY GROWN TREE (no depth limit) ===")
print(f"Train F1: {train_f1_full*100:.1f}%")
print(f"Test  F1: {test_f1_full*100:.1f}%")
print(f"Tree depth: {tree_full.get_depth()}")
print()
print("💡 Notice: Train F1 is very high but Test F1 is lower.")
print("   This gap = OVERFITTING. The tree memorized training data.")


# --- Fix overfitting: limit the tree depth ---
# max_depth controls how many questions the tree can ask

results = []
for depth in [2, 3, 5, 7, 10, 15, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    results.append({
        "max_depth":  str(depth),
        "train_f1":  f1_score(y_train, tree.predict(X_train)),
        "test_f1":   f1_score(y_test,  tree.predict(X_test)),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# Plot
plt.figure(figsize=(9, 4))
plt.plot(results_df["max_depth"], results_df["train_f1"], marker="o",
         label="Train F1", color="#6366f1")
plt.plot(results_df["max_depth"], results_df["test_f1"],  marker="o",
         label="Test F1",  color="#ef4444")
plt.title("Overfitting: Train vs Test F1 at Different Depths")
plt.xlabel("max_depth")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# 💡 The gap between train and test lines = overfitting gap
# A good depth = where test F1 is highest



# ✅ Visualize the tree at depth=3 — see exactly what it learned

tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)

plt.figure(figsize=(20, 8))
plot_tree(
    tree_shallow,
    feature_names=X.columns.tolist(),
    class_names=["Not Viral", "Viral"],
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Decision Tree (depth=3) — Each node shows the question asked", fontsize=14)
plt.show()
# 💡 Read top-down. Each box shows:
# - The question (e.g. shares <= 234)
# - How many samples reach this node
# - The predicted class at this point



rf_model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)

rf_model.fit(X_train, y_train)

rf_train_f1 = f1_score(y_train, rf_model.predict(X_train))
rf_test_f1  = f1_score(y_test,  rf_model.predict(X_test))

print("=== RANDOM FOREST (100 trees) ===")
print(f"Train F1: {rf_train_f1*100:.1f}%")
print(f"Test  F1: {rf_test_f1*100:.1f}%")
print()
print("=== COMPARISON ===")
print(f"Single Tree (full)  — Train: {train_f1_full*100:.1f}%  Test: {test_f1_full*100:.1f}%")
print(f"Single Tree (d=3)   — Train: {f1_score(y_train, tree_shallow.predict(X_train))*100:.1f}%  Test: {f1_score(y_test, tree_shallow.predict(X_test))*100:.1f}%")
print(f"Random Forest       — Train: {rf_train_f1*100:.1f}%  Test: {rf_test_f1*100:.1f}%")



# ✅ Feature Importance

importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("=== FEATURE IMPORTANCE (Random Forest) ===")
for feat, score in importances.items():
    bar = "█" * int(score * 100)
    print(f"  {feat:20s}: {score:.4f}  {bar}")

# Plot
plt.figure(figsize=(9, 5))
importances.plot(kind="barh", color="#10b981")
plt.title("Feature Importance — What drives viral posts?")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.show()




from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score


# Logistic Regression needs scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Decision Tree
best_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
best_tree.fit(X_train, y_train)
tree_pred = best_tree.predict(X_test)

# Random Forest (already trained)
rf_pred = rf_model.predict(X_test)

# Compare
models = {
    "Logistic Regression": lr_pred,
    "Decision Tree (d=5)": tree_pred,
    "Random Forest":       rf_pred,
}

print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 70)
for name, pred in models.items():
    print(f"{name:<25} "
          f"{accuracy_score(y_test, pred)*100:>9.1f}% "
          f"{precision_score(y_test, pred)*100:>9.1f}% "
          f"{recall_score(y_test, pred)*100:>9.1f}% "
          f"{f1_score(y_test, pred)*100:>9.1f}%")
    
    
    
# ✅ Predict on brand new posts

new_posts = pd.DataFrame({
    "likes":          [12000, 300,  5000],
    "comments":       [890,   15,   420],
    "shares":         [650,   8,    180],
    "hashtag_count":  [5,     25,   10],
    "caption_length": [120,   8,    200],
    "hour_posted":    [18,    3,    12],
    "follower_count": [50000, 800,  12000],
    "is_video":       [1,     0,    1],
})

predictions  = rf_model.predict(new_posts)
probabilities = rf_model.predict_proba(new_posts)

post_names = ["Post A (high engagement)", "Post B (low engagement)", "Post C (medium)"]

print("=== VIRAL PREDICTION RESULTS ===")
for name, pred, prob in zip(post_names, predictions, probabilities):
    label = "🔥 VIRAL" if pred == 1 else "❌ NOT VIRAL"
    print(f"{label}  ({prob[1]*100:.1f}% viral probability)")
    print(f"   {name}")
    print()
