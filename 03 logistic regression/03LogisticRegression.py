import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 200)
y = sigmoid(x)




plt.figure(figsize=(8, 4))
plt.plot(x, y, color="#6366f1", linewidth=2)
plt.axhline(y=0.5, color="red", linestyle="--", label="Decision boundary (0.5)")
plt.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
plt.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
plt.title("Sigmoid Function — squashes any number into 0 to 1")
plt.xlabel("Weighted sum of features")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Left of 0 → probability < 0.5 → predict NOT spam
# Right of 0 → probability > 0.5 → predict SPAM

np.random.seed(42)
n = 1000

# --- Generate legitimate posts (label = 0) ---
n_legit = 800
legit = pd.DataFrame({
    "likes":          np.random.randint(50, 10000, n_legit),
    "comments":       np.random.randint(10, 1000, n_legit),
    "shares":         np.random.randint(5, 500, n_legit),
    "caption_length": np.random.randint(20, 300, n_legit),
    "hashtag_count":  np.random.randint(1, 10, n_legit),
    "hour_posted":    np.random.randint(6, 23, n_legit),
    "is_spam":        0
})



# --- Generate spam posts (label = 1) ---
n_spam = 200
spam = pd.DataFrame({
    "likes":          np.random.randint(5000, 50000, n_spam),  # inflated likes
    "comments":       np.random.randint(0, 20, n_spam),        # almost no comments
    "shares":         np.random.randint(1000, 5000, n_spam),   # bot shares
    "caption_length": np.random.randint(1, 15, n_spam),        # very short
    "hashtag_count":  np.random.randint(15, 30, n_spam),       # hashtag stuffing
    "hour_posted":    np.random.randint(1, 5, n_spam),         # odd hours
    "is_spam":        1
})


# Combine and shuffle
df = pd.concat([legit, spam], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df["is_spam"].value_counts())
print(f"\nSpam rate: {df['is_spam'].mean()*100:.1f}%")


# ✅ Explore differences between spam and legit posts visually

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Spam vs Legitimate Posts — Feature Comparison", fontsize=14, fontweight="bold")

features = ["likes", "comments", "shares", "caption_length", "hashtag_count", "hour_posted"]
colors = {0: "#6366f1", 1: "#ef4444"}
labels = {0: "Legitimate", 1: "Spam"}

for ax, feature in zip(axes.flatten(), features):
    for label in [0, 1]:
        data = df[df["is_spam"] == label][feature]
        ax.hist(data, bins=30, alpha=0.6, color=colors[label], label=labels[label])
    ax.set_title(feature)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()



df

x= df.drop('is_spam' , axis = 1 )
y= df['is_spam']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)




# Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled , y_train )



y_pred = model.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("=" * 45)
print("       SPAM DETECTOR EVALUATION")
print("=" * 45)
print(f"Accuracy:  {accuracy*100:.1f}%  — overall correct predictions")
print(f"Precision: {precision*100:.1f}%  — of flagged spam, actually spam")
print(f"Recall:    {recall*100:.1f}%  — of all spam, model caught this %")
print(f"F1 Score:  {f1*100:.1f}%  — balance of precision + recall")

print("\n--- Full Report ---")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Spam"]))

# ✅ Confusion Matrix — the most informative classification visual
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nLegit", "Predicted\nSpam"])
ax.set_yticklabels(["Actual\nLegit", "Actual\nSpam"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else "black")

ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.colorbar(im)
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives  (correctly said legit): {tn}")
print(f"False Positives (wrongly flagged as spam): {fp}")
print(f"False Negatives (missed spam): {fn}")
print(f"True Positives  (correctly caught spam): {tp}")



proba = model.predict_proba(X_test_scaled)

results = pd.DataFrame({
    "actual":       y_test.values[:8],
    "predicted":    y_pred[:8],
    "prob_legit":   proba[:8, 0].round(3),
    "prob_spam":    proba[:8, 1].round(3),
})
print(results.to_string(index=False))


new_posts = pd.DataFrame({
    "likes":          [250,   45000, 1200],
    "comments":       [80,    3,     340],
    "shares":         [30,    2200,  90],
    "caption_length": [180,   5,     120],
    "hashtag_count":  [4,     28,    6],
    "hour_posted":    [14,    3,     19],
})

new_posts_scaled = scaler.transform(new_posts)
predictions = model.predict(new_posts_scaled)
probabilities = model.predict_proba(new_posts_scaled)

post_labels = ["Post A (looks legit)", "Post B (looks spammy)", "Post C (normal)"]

print("=== SPAM DETECTION RESULTS ===")
for label, pred, prob in zip(post_labels, predictions, probabilities):
    result = "🚨 SPAM" if pred == 1 else "✅ LEGIT"
    print(f"{label}")
    print(f"  → {result} (spam probability: {prob[1]*100:.1f}%)")
    

