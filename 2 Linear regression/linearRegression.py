import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



np.random.seed(42)
n = 200

df = pd.DataFrame({
    "likes":    np.random.randint(100, 20000, n),
    "comments": np.random.randint(10, 2000, n),
    "shares":   np.random.randint(5, 1500, n),
})


df['engagement'] = df['likes'] + df['shares'] + df['comments']


df.isnull().sum()


plt.figure(figsize=(8, 5))
plt.scatter(df["likes"], df["engagement"], alpha=0.5, color="#6366f1")
plt.title("Likes vs Engagement (before training)")
plt.xlabel("Likes")
plt.ylabel("Total Engagement")
plt.grid(True, alpha=0.3)
plt.show()



x = df.drop("engagement" , axis = 1)
y= df['engagement']

X_train, X_test, y_train, y_test  = train_test_split(x, y , test_size=0.2, random_state=42)


model = LinearRegression() 
model.fit(X_train, y_train)

print("✅ Model trained!")
print(f"\nWeights (one per feature): {model.coef_}")
print(f"  likes weight:    {model.coef_[0]:.4f}")
print(f"  comments weight: {model.coef_[1]:.4f}")
print(f"  shares weight:   {model.coef_[2]:.4f}")
print(f"\nBias (intercept): {model.intercept_:.4f}")


y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual":    y_test.values[:10],
    "Predicted": y_pred[:10].round(0),
    "Difference": (y_test.values[:10] - y_pred[:10]).round(0)
})

print(results.to_string(index=False))


mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=== MODEL EVALUATION ===")
print(f"MAE:   {mae:.1f}")
print(f"RMSE:  {rmse:.1f}")
print(f"R²:    {r2:.4f}")

print(f"\n📊 Interpretation:")
print(f"  On average, predictions are off by {mae:.0f} engagement points")
print(f"  The model explains {r2*100:.1f}% of the variation in engagement")

# ✅ Visualize: Actual vs Predicted
# Perfect model = all dots on the diagonal line

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="#6366f1")

# Draw the perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

plt.title("Actual vs Predicted Engagement")
plt.xlabel("Actual Engagement")
plt.ylabel("Predicted Engagement")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ✅ Predict on a brand new post — this is the real use case!
# Someone creates a post. Before it goes viral, can we predict engagement?

new_post = pd.DataFrame({
    "likes":    [3500],
    "comments": [280],
    "shares":   [120]
})


predicted_engagement = model.predict(new_post)[0]

print(f"Predicted total engagement: {predicted_engagement:,.0f}")
print(f"Actual (manual): {3500 + 280 + 120}")



x2 = df[['likes']]
y2 = df['engagement']

X_train2, X_test2, y_train2, y_test2  = train_test_split(x2, y2 , test_size=0.2, random_state=42)


model2=LinearRegression()



model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)



newR2   = r2_score(y_test2, y_pred2)







