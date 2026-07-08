import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score ,GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Viral posts: {y.sum()} ({y.mean()*100:.1f}%)")






decisionTree = DecisionTreeClassifier(random_state=42 , max_depth=5)



cv_score_decisionTree = cross_val_score(decisionTree, X, y, cv=5, scoring='f1')

rf_model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores — trains faster
)



cv_scoreRandomForest = cross_val_score(rf_model, X, y, cv=5, scoring='f1')


print(f"\n  Mean F1:  {cv_score_decisionTree.mean()*100:.2f}%")
print(f"  Std Dev:  {cv_score_decisionTree.std()*100:.2f}%")
print()
print(f"\n  Mean F1:  {cv_scoreRandomForest.mean()*100:.2f}%")
print(f"  Std Dev:  {cv_scoreRandomForest.std()*100:.2f}%")








# Step 1: Build the pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),       # which scaler?
    ('model',  LogisticRegression())        # which model?
])

# Step 2: Define param grid
# Note: prefix with 'model__' because it's inside a pipeline
param_grid = {
    'model__C': [0.01, 0.1, 1, 10]    # C = regularization strength
}

# Step 3: GridSearchCV
grid = GridSearchCV(
    estimator=lr_pipeline,         # pass the pipeline
    param_grid=param_grid,        # pass the param grid
    cv=5,
    scoring='f1',
    verbose=1
)

# Step 4: Fit and print results
grid.fit(X_train, y_train)
print(f"Best C value: {grid.best_params_}")
print(f"Best CV F1:   {grid.best_score_*100:.2f}%")



def evaluate_model(pipeline, X, y, threshold=0.80):
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    
    mean = scores.mean()
    std  = scores.std()
    
    print(f"Mean F1: {mean*100:.2f}% ± {std*100:.2f}%")
    
    if mean >= threshold:
        print(f"Status: ✅ PASS")
    else:
        print(f"Status: ❌ FAIL")
    
    return mean