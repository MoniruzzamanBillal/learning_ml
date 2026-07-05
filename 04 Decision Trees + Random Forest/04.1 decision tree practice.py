
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


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



x = df.drop("is_viral", axis=1)
y = df["is_viral"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



rf_model1 = RandomForestClassifier(
    n_estimators=50,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)

rf_model2 = RandomForestClassifier(
    n_estimators=10,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)

rf_model3 = RandomForestClassifier(
    n_estimators=200,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)




rf_model1.fit(x_train, y_train)
rf_model2.fit(x_train, y_train)
rf_model3.fit(x_train, y_train)



model1F1 = f1_score(y_test, rf_model1.predict(x_test))
model2F1 = f1_score(y_test, rf_model2.predict(x_test))
model3F1 = f1_score(y_test, rf_model3.predict(x_test))

print(f"Model 1 F1 Score: {model1F1}")
print(f"Model 2 F1 Score: {model2F1}")
print(f"Model 3 F1 Score: {model3F1}")




rf_model4 = RandomForestClassifier(
    n_estimators=50,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
    , max_depth=5 
)



rf_model5 = RandomForestClassifier(
    n_estimators=50,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
   
)


rf_model4.fit(x_train, y_train)
rf_model5.fit(x_train, y_train)


model4F1 = f1_score(y_test, rf_model4.predict(x_test))
model5F1 = f1_score(y_test, rf_model5.predict(x_test))


print(f"Model 4 F1 Score: {model4F1}")
print(f"Model 5 F1 Score: {model5F1}")




df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["follower_count"] + 1)
df["shares_per_like"] = df["shares"] / (df["likes"] + 1)

x2 = df.drop("is_viral", axis=1)
y2 = df["is_viral"]

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)



rf_new = RandomForestClassifier(
    n_estimators=50,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)

rf_new.fit(x2_train, y2_train)

newF1 = f1_score(y2_test, rf_new.predict(x2_test))

print(f"New Model F1 Score: {newF1}")



# ✅ Feature Importance

importances = pd.Series(
    rf_new.feature_importances_,
    index=x2.columns
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





