import numpy as np
import pandas as pd


# Likes for 8 posts (regular Python list)
likes_list = [4823, 921, 15200, 632, 2100, 3400, 8900, 5600]

# Convert to a NumPy array
likes = np.array(likes_list)

print(f"Type: {type(likes)}")
print(f"Array: {likes}")
print(f"Shape: {likes.shape}") 



# ✅ The power of NumPy — math on the whole array at once
# With a normal Python list, you'd need:
doubled_list = [x * 2 for x in likes_list]

doubledArray = likes * 2 
print(f"Doubled: {doubledArray}")

bonus = likes + 100 
print(f"bonus: {bonus}")

# Built-in statistics — no manual loops needed!

print(f"Mean (average): {np.mean(likes)}")
print(f"Median: {np.median(likes)}")
print(f"Standard deviation: {np.std(likes):.2f}")
print(f"sum : {np.sum(likes)}")


# ✅ Boolean filtering — extremely common in ML data cleaning
mask = likes > 3000 
print(f"Mask (True/False): {mask}")

high_performing_likes = likes[mask]
print(f"High performing posts (likes): {high_performing_likes}")



comments = np.array([312, 445, 870, 98, 180, 560, 430, 920])
shares = np.array([89, 230, 1200, 45, 60, 890, 670, 340])

# This is called 'boolean indexing' — you'll use this CONSTANTLY in ML
# Combine multiple arrays into a calculation
totalEngagement = likes + comments + shares

print(f"\nTotal engagement per post: {totalEngagement}")
print(f"Average engagement: {np.mean(totalEngagement):.2f}")



data = [
    {"platform": "Instagram", "likes": 4823,  "comments": 312, "shares": 89,   "category": "tech"},
    {"platform": "Twitter",   "likes": 921,   "comments": 445, "shares": 230,  "category": "news"},
    {"platform": "TikTok",    "likes": 15200, "comments": 870, "shares": 1200, "category": "tech"},
    {"platform": "LinkedIn",  "likes": 632,   "comments": 98,  "shares": 45,   "category": "career"},
    {"platform": "Instagram", "likes": 2100,  "comments": 180, "shares": 60,   "category": "tech"},
    {"platform": "Twitter",   "likes": 3400,  "comments": 560, "shares": 890,  "category": "news"},
    {"platform": "TikTok",    "likes": 8900,  "comments": 430, "shares": 670,  "category": "entertainment"},
    {"platform": "YouTube",   "likes": 5600,  "comments": 920, "shares": 340,  "category": "tech"},
    {"platform": "LinkedIn",  "likes": 1200,  "comments": 210, "shares": 88,   "category": "career"},
    {"platform": "YouTube",   "likes": 9800,  "comments": 1200,"shares": 560,  "category": "entertainment"},
]

dataFrame = pd.DataFrame(data)

dataFrame.head()




print("=== Shape (rows, columns) ===")
print(dataFrame.shape)

print("\n=== Column names ===")
print(dataFrame.columns.tolist())



print("\n=== Quick statistics summary ===")
dataFrame.describe()


likesColumn  = dataFrame['likes']

print(likesColumn)

subSets = dataFrame[['likes', 'comments']]
print(subSets)

print("\n--- Select rows by condition (like SQL WHERE / Prisma findMany where) ---")

techPosts = dataFrame[dataFrame['category'] == 'tech']
print(techPosts)


# ✅ Creating new columns (feature engineering — a core ML skill!)

dataFrame["engagement"] = dataFrame['likes']  + dataFrame['comments'] + dataFrame['shares']

dataFrame['isViral'] = dataFrame['engagement'] > 1000 


dataFrame.head()


# ✅ Sorting — like ORDER BY in SQL 
topPosts = dataFrame.sort_values('engagement' , ascending=False)
topPosts.head()

# ✅ Group By — exactly like SQL GROUP BY / Prisma groupBy
# This is one of the MOST used Pandas operations

platformAverage = dataFrame.groupby('platform')['engagement'].mean().sort_values(ascending=False) 

print("=== Average Engagement by Platform ===")
print(platformAverage)

messy_data = [
    {"platform": "Instagram", "likes": 4823,  "comments": 312,  "shares": 89},
    {"platform": "Twitter",   "likes": np.nan,"comments": 445,  "shares": 230},   # missing likes
    {"platform": "TikTok",    "likes": 15200, "comments": np.nan,"shares": 1200}, # missing comments
    {"platform": "LinkedIn",  "likes": 632,   "comments": 98,   "shares": np.nan},# missing shares
    {"platform": "Instagram", "likes": 2100,  "comments": 180,  "shares": 60},
]

myDataFrame = pd.DataFrame(messy_data)

print(myDataFrame)


print("\n=== Check for missing values ===")
print(myDataFrame.isnull().sum())




# ✅ Two common strategies for missing data

droppedDataframe = myDataFrame.dropna()
print("=== After dropping missing rows ===")
print(droppedDataframe)




# Strategy 2: Fill missing values (often better — keeps more data)
filledData = myDataFrame.copy() 

filledData['likes'] = filledData['likes'].fillna(filledData['likes'].mean())
filledData['comments'] = filledData['comments'].fillna(filledData['comments'].mean()) 
filledData['shares'] = filledData['shares'].fillna(filledData['shares'].mean())




print()
print(filledData)

