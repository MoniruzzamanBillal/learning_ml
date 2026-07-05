# // ---
# // ## 🏋️ Section 7: Mini Project — Social Media Analyzer

# // Now put everything together. This is your **Week 1-2 project.**

# // You'll build a small analyzer that:
# // 1. Takes a dataset of posts
# // 2. Calculates stats per platform
# // 3. Identifies viral posts
# // 4. Prints a clean report

# // > This is **exactly** the kind of thinking you'll use in real ML pipelines — load data, analyze it, extract insights.




posts = [
    {"id": 1,  "platform": "Instagram", "likes": 4823,  "comments": 312, "shares": 89,   "category": "tech"},
    {"id": 2,  "platform": "Twitter",   "likes": 921,   "comments": 445, "shares": 230,  "category": "news"},
    {"id": 3,  "platform": "TikTok",    "likes": 15200, "comments": 870, "shares": 1200, "category": "tech"},
    {"id": 4,  "platform": "LinkedIn",  "likes": 632,   "comments": 98,  "shares": 45,   "category": "career"},
    {"id": 5,  "platform": "Instagram", "likes": 2100,  "comments": 180, "shares": 60,   "category": "tech"},
    {"id": 6,  "platform": "Twitter",   "likes": 3400,  "comments": 560, "shares": 890,  "category": "news"},
    {"id": 7,  "platform": "TikTok",    "likes": 8900,  "comments": 430, "shares": 670,  "category": "entertainment"},
    {"id": 8,  "platform": "YouTube",   "likes": 5600,  "comments": 920, "shares": 340,  "category": "tech"},
    {"id": 9,  "platform": "LinkedIn",  "likes": 1200,  "comments": 210, "shares": 88,   "category": "career"},
    {"id": 10, "platform": "YouTube",   "likes": 9800,  "comments": 1200,"shares": 560,  "category": "entertainment"},
]



VIRAL_THRESHOLD = 10000


for post in posts:
  post["engagement"] = post["likes"] + post["comments"] + post["shares"]
  post["is_viral"] = post["engagement"] >= VIRAL_THRESHOLD


print(posts)

viralPosts = [post for post in posts if post['is_viral']]

print("Viral Posts")

for post in viralPosts:
    print(
        f"ID: {post['id']}, "
        f"Platform: {post['platform']}, "
        f"Engagement: {post['engagement']}"
    )
