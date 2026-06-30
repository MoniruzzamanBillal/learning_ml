import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression



myData = pd.read_csv("/home/moniruzzaman-billal/Work/own_Learning/3ML/ref Data/nasdaq100.csv" , sep=";")

myData.head()

myData.isnull().sum()

myData = myData.drop(columns= ["Date"] )

plt.scatter(myData['Starting (USD)'] , myData['Ending (USD)'] )
plt.xlabel("Starting")
plt.ylabel("Ending")
plt.title("basic nasdaq data!!")


x = myData.drop(columns=['Ending (USD)']  )
x.head()

y= myData.drop(columns=['Starting (USD)']  )
y.head()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x , y)

reg.predict([[16800]]) 


myData['predect_Y'] = reg.predict(x)
myData.head()



plt.scatter( x.mean() ,y.mean() , color="red" )
plt.scatter(myData['Starting (USD)'] , myData['Ending (USD)'] )
plt.xlabel("Starting")
plt.ylabel("Ending")
plt.plot(x , reg.predict(x) , color="blue")

plt.show()


















