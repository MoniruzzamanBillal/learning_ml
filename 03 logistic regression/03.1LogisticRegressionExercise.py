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
n = 1000

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

n_spam = 200
spam = pd.DataFrame({
    "likes":          np.random.randint(5000, 50000, n_spam), 
    "comments":       np.random.randint(0, 20, n_spam),        
    "shares":         np.random.randint(1000, 5000, n_spam),   
    "caption_length": np.random.randint(1, 15, n_spam),        
    "hashtag_count":  np.random.randint(15, 30, n_spam),       
    "hour_posted":    np.random.randint(1, 5, n_spam),         
    "is_spam":        1
})


df = pd.concat([legit, spam], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



x= df.drop('is_spam' , axis = 1 )
y= df['is_spam']


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

scaler = StandardScaler()

xTrainScaled = scaler.fit_transform(x_train)
xTestScaled = scaler.transform(x_test)

model = LogisticRegression(random_state=42)



model.fit(xTrainScaled , y_train)


proba = model.predict_proba(xTestScaled)
pred = model.predict(xTestScaled)



newX = df[[ "comments","hashtag_count", "hour_posted"]]
newY = df["is_spam"]

newX_train , newX_test , newY_train , newY_test = train_test_split(newX , newY , test_size=0.2 , random_state=42)

newXTrainScaled = scaler.fit_transform(newX_train)
newXTestScaled = scaler.transform(newX_test)

newModel = LogisticRegression(random_state=42)

newModel.fit(newXTrainScaled , newY_train)

predictedValue = newModel.predict(newXTestScaled)


accuracy  = accuracy_score(newY_test, predictedValue)
precision = precision_score(newY_test, predictedValue)
recall    = recall_score(newY_test, predictedValue)
f1        = f1_score(newY_test, predictedValue)

print(f"Accuracy:  {accuracy*100:.1f}%  — overall correct predictions")
print(f"Precision: {precision*100:.1f}%  — of flagged spam, actually spam")
print(f"Recall:    {recall*100:.1f}%  — of all spam, model caught this %")
print(f"F1 Score:  {f1*100:.1f}%  — balance of precision + recall")




newData = df.copy()


newData['likeToCommentRation'] = newData['likes'] / (newData['comments'] + 1)


xData = newData.drop('is_spam', axis=1)
yData = newData['is_spam']

xData_train, xData_test, yData_train, yData_test = train_test_split(xData, yData, test_size=0.2, random_state=42)

scaler = StandardScaler()

xData_train_scaled = scaler.fit_transform(xData_train)
xData_test_scaled = scaler.transform(xData_test)

model = LogisticRegression(random_state=42)


model.fit(xData_train_scaled, yData_train)


newPredictedValue = model.predict(xData_test_scaled)

accuracy  = accuracy_score(yData_test, newPredictedValue)
precision = precision_score(yData_test, newPredictedValue)
recall    = recall_score(yData_test, newPredictedValue)
f1        = f1_score(yData_test, newPredictedValue)

print(f"Accuracy:  {accuracy*100:.1f}%  — overall correct predictions")
print(f"Precision: {precision*100:.1f}%  — of flagged spam, actually spam")
print(f"Recall:    {recall*100:.1f}%  — of all spam, model caught this %")
print(f"F1 Score:  {f1*100:.1f}%  — balance of precision + recall")

