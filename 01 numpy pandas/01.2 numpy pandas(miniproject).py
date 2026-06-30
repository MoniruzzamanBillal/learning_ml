
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42) 


platforms = ["Instagram", "Twitter", "TikTok", "LinkedIn", "YouTube"]
categories = ["tech", "news", "entertainment", "career", "lifestyle"]

n_posts = 200


dataset = pd.DataFrame({
    "post_id": range(1, n_posts + 1),
    "platform": np.random.choice(platforms, n_posts),
    "category": np.random.choice(categories, n_posts),
    "likes": np.random.randint(100, 20000, n_posts),
    "comments": np.random.randint(10, 2000, n_posts),
    "shares": np.random.randint(5, 1500, n_posts),
})


dataset.to_csv("social_media_data.csv", index=False)
print("✅ Dataset saved as social_media_data.csv")
dataset.head()




df = pd.read_csv("social_media_data.csv")

df["engagement"] = df["likes"] + df["comments"] + df["shares"]
df["is_viral"] = df["engagement"] >= df["engagement"].quantile(0.90) 

print("="*60)
print("           SOCIAL MEDIA DATASET ANALYSIS REPORT")
print("="*60)

print(f"\n📊 Total posts analyzed: {len(df)}")
print(f"🔥 Viral posts: {df['is_viral'].sum()} ({df['is_viral'].mean()*100:.1f}%)")

print("\n--- Engagement by Platform ---")
platform_report = df.groupby("platform").agg(
    avg_engagement=("engagement", "mean"),
    total_posts=("post_id", "count"),
    viral_count=("is_viral", "sum")
).round(0).sort_values("avg_engagement", ascending=False)
print(platform_report)

print("\n--- Engagement by Category ---")
category_report = df.groupby("category")["engagement"].mean().round(0).sort_values(ascending=False)
print(category_report)

print("\n--- Top 5 Most Viral Posts ---")
top5 = df.sort_values("engagement", ascending=False).head(5)
print(top5[["post_id", "platform", "category", "engagement"]].to_string(index=False))

print("\n" + "="*60)
best_platform = platform_report.index[0]
best_category = category_report.index[0]
print(f"🏆 Best performing platform: {best_platform}")
print(f"🏆 Best performing category: {best_category}")
print("="*60)



platform_report["avg_engagement"].plot(kind="bar", figsize=(8,5), color="#6366f1")
plt.title("Average Engagement by Platform")
plt.ylabel("Average Engagement")
plt.xlabel("Platform")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()