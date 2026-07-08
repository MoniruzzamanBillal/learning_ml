import matplotlib.pyplot as plt

# --- Line chart: engagement over time ---
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
engagement = [1200, 1900, 1500, 2400, 3200, 4100, 3800]

plt.figure(figsize=(8, 5))
plt.plot(days, engagement, marker="o", color="#6366f1", linewidth=2)
plt.title("Weekly Engagement Trend")
plt.xlabel("Day of Week")
plt.ylabel("Total Engagement")
plt.grid(True, alpha=0.3)
plt.show()

# --- Bar chart: comparing platforms ---
platforms = ["Instagram", "Twitter", "TikTok", "LinkedIn", "YouTube"]
avg_engagement = [3200, 2100, 8900, 950, 4500]

plt.figure(figsize=(8, 5))
bars = plt.bar(platforms, avg_engagement, color="#10b981")
plt.title("Average Engagement by Platform")
plt.xlabel("Platform")
plt.ylabel("Avg Engagement")

# Add value labels on top of bars — looks more professional
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 50, f"{height:,.0f}",
             ha="center", fontsize=9)

plt.show()

# --- Scatter plot: relationship between two variables ---
# Does more comments mean more shares? Let's check visually.

import numpy as np
np.random.seed(1)

comments = np.random.randint(50, 1000, 50)
shares = comments * 0.8 + np.random.randint(-100, 100, 50)  # roughly correlated

plt.figure(figsize=(8, 5))
plt.scatter(comments, shares, color="#f59e0b", alpha=0.6)
plt.title("Comments vs Shares")
plt.xlabel("Comments")
plt.ylabel("Shares")
plt.grid(True, alpha=0.3)
plt.show()

# 💡 Scatter plots are how you visually spot correlations BEFORE
# building a model — this is literally a preview of Linear Regression in Phase 2!


# --- Pie chart: category distribution ---
categories = ["tech", "news", "entertainment", "career", "lifestyle"]
post_counts = [45, 30, 60, 20, 35]

plt.figure(figsize=(7, 7))
plt.pie(post_counts, labels=categories, autopct="%1.1f%%",
        colors=["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"])
plt.title("Post Distribution by Category")
plt.show()


# --- Multiple charts side by side using subplots ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left chart: bar
axes[0].bar(platforms, avg_engagement, color="#6366f1")
axes[0].set_title("Avg Engagement by Platform")
axes[0].set_ylabel("Engagement")
axes[0].tick_params(axis="x", rotation=30)

# Right chart: line
axes[1].plot(days, engagement, marker="o", color="#10b981")
axes[1].set_title("Weekly Engagement Trend")
axes[1].set_ylabel("Engagement")

plt.tight_layout()
plt.show()
