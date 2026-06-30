


platformName = "Twitter!!"

print(platformName)
print(f"Platform: {platformName}")

platformName = 55

print(platformName)


platforms = ["Facebook", "YouTube", "Instagram", "TikTok", "Twitter", "LinkedIn", "Snapchat"]
users_millions = [2900, 2500, 2000, 1500, 550, 900, 750]

print(f"first platform = {platforms[0]}")
print(f"last platform = {platforms[-1]}")
print(f"first 3 platforms = {platforms[0:3]}")
print(f"total platforms = {len(platforms)}")


# Adding to a list  (JS: .push())
platforms.append("Pinterest")

print(f"all platforms = {platforms}")


for platform in platforms:
  print(f"single platform = {platform}")

print()

# Loop with index — use enumerate() (like JS .forEach with index)
for index , platform in enumerate(platforms):
  print(f"index = {index} , platform = {platform}")



# --- Example 1: Filter ---
# Get platforms with more than 1000 million users

# JS way: users_millions.filter(u => u > 1000)
# Python way:
filteredUsers = [u for u in users_millions if u > 1000]

print(f"filteredUsers = {filteredUsers}")


# --- Example 2: Filter with paired lists ---

lergePlatformUsers = [i for u , i in zip(users_millions, platforms) if u > 1000 ]

print(f"largerPlatformUsers = {lergePlatformUsers}")

# --- Example 3: Transform (map) ---
convertedUsers = [u*5 for u in users_millions if u > 1000]

print(f"converted users = {convertedUsers} ")

# --- Example 4: Filter + Transform together ---
smallPlatforms = [p.upper() for p , u in zip(platforms , users_millions) if u < 1000]


print(f"smallPlatforms = {smallPlatforms}")





# ✅ SECTION 4: Dictionaries

# A social media post represented as a dictionary
post = {
    "id": 1001,
    "platform": "Instagram",
    "content": "Just launched my new app! 🚀",
    "likes": 4823,
    "comments": 312,
    "shares": 89,
    "is_viral": True
}

# Accessing values  (JS: post.likes or post['likes'])
print(f"platform = {post['platform']}")
print(f"likes = {post['likes']}")


# Adding a new key
post['totalEngagement'] = post['likes'] + post['comments'] + post['shares']

print(f"totalEngagement = {post['totalEngagement']}")

# Looping through a dict
print("\n=== Post Details ===")
for key , value in post.items():
  print(f"{key} : {value}")



# ✅ List of Dictionaries — this is how REAL datasets look!
# Think of this like an array of objects in JS

posts = [
    {"id": 1, "platform": "Instagram", "likes": 4823, "comments": 312, "shares": 89, "category": "tech"},
    {"id": 2, "platform": "Twitter",   "likes": 921,  "comments": 445, "shares": 230, "category": "news"},
    {"id": 3, "platform": "TikTok",    "likes": 15200,"comments": 870, "shares": 1200,"category": "tech"},
    {"id": 4, "platform": "LinkedIn",  "likes": 632,  "comments": 98,  "shares": 45, "category": "career"},
    {"id": 5, "platform": "Instagram", "likes": 2100, "comments": 180, "shares": 60, "category": "tech"},
    {"id": 6, "platform": "Twitter",   "likes": 3400, "comments": 560, "shares": 890, "category": "news"},
    {"id": 7, "platform": "TikTok",    "likes": 8900, "comments": 430, "shares": 670, "category": "entertainment"},
    {"id": 8, "platform": "YouTube",   "likes": 5600, "comments": 920, "shares": 340, "category": "tech"},
]


posts = [{  **post,  "engagement" : post['likes'] + post['shares'] + post['comments'] } for post in posts ]

# alternative 
for post in posts:
  post["engagement2"] = post['likes'] + post['shares'] + post['comments']

print(f"postWithEngagement = {posts}")


allLikes = [post['likes'] for post in posts]

print(f"allLikes = {allLikes}")


totalLikes = sum(allLikes)
count  = len(allLikes)
averageLikes = totalLikes / count
maxLikes = max(allLikes)
minLikes = min(allLikes)

print(f"count = {count}")
print(f"averageLikes = {averageLikes}")
print(f"totalLikes = {totalLikes}")
print(f"maxLikes = {maxLikes}")
print(f"minLikes = {minLikes}")





mostViralPost = max(posts , key=lambda post:post['engagement'])

print(f"mostViralPost = {mostViralPost}")

sortedPost = sorted(posts , key=lambda post:post['engagement'] , reverse=True )

print(f"sortedPost = {sortedPost}")





def calculateStats(values, label="Values"):
  total = sum(values)
  count = len(values)
  average = total / count
  maxVal = max(values)
  minVal = min(values)
  print(f"\n=== {label} Stats ===")
  print(f"count = {count}")
  print(f"average = {average}")
  print(f"total = {total}")
  print(f"max = {maxVal}")
  print(f"min = {minVal}")


def getPostByPlatform(values , platform ):
  platformData = [value for value in values if value['platform'] == platform ]
  return platformData

def engagement_rate(post):
  return post["likes"] + (post["comments"] * 2) + (post["shares"] * 3)




allLikes= [post['likes'] for post in posts]

statsValue =  calculateStats(allLikes, "Likes")

InstagramPost = getPostByPlatform(posts , "Instagram")
instapost = getPostByPlatform(posts , "Instagram")

print(f"InstagramPost = {InstagramPost}")


for post in posts:
  post["weighted_engagement"] = engagement_rate(post)


print(f"posts = {posts}")




