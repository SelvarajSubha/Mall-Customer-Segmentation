import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

 df=pd.read_csv('Customer Dataset.csv')

 df.head(12)

 df.shape

 df.isnull().sum()

 x=df.iloc[:,[7,10]].values

sns.set()
sns.countplot(x=df.Entertainment,data=df)
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
from sklearn.cluster import AgglomerativeClustering
      hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
      y_hc=hc.fit_predict(x)
y_hc
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='blue',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='green',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='red',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='yellow',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='black',label='Cluster 5')
plt.title("Clusters of customers")
plt.xlabel('Clothing')
plt.ylabel('Spending score of Clothing')
plt.legend()
plt.show()

plt.scatter(x[...,0],x[...,1])
plt.show()
df['Target']=y_hc
df
sns.set()
sns.countplot(x=df.Gender,data=df)
plt.show()
sns.set()
sns.countplot(x=df.Clothing,data=df)
plt.show()
sns.set()
sns.countplot(x=df.Food,data=df)
plt.show()
sns.set()
sns.countplot(x=df.Entertainment,data=df)
plt.show()
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
df1=df[['Spending score of Food','Spending score of Clothing']]
df1.head()

sns.scatterplot(df1['Spending score of Food'],df1['Spending score of Clothing'])


Clothing=df.groupby('Customer ID')['Clothing'].count()
Clothing=Clothing.reset_index()
Clothing.head()

sns.distplot(df['Spending score of Food'])
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='black',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='blue',label='Cluster 5')

plt.title("Clusters of customers")
plt.xlabel("Food")
plt.ylabel("Spending score of food")
plt.legend()
plt.show()

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='black',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='blue',label='Cluster 5')

plt.title("Clusters of customers")
plt.xlabel("Entertainment")
plt.ylabel("Spending score of Entertainment")
plt.legend()
plt.show()

  
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='blue',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='green',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='red',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='yellow',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='black',label='Cluster 5')

plt.title("Clusters of customers")
plt.xlabel('Footwear')
plt.ylabel('Spending score of Footwear')
plt.legend()
plt.show()

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='blue',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='green',label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='red',label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='yellow',label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='black',label='Cluster 5')

plt.title("Clusters of customers")
plt.xlabel('Footwear')
plt.ylabel('Spending score of Footwear')
plt.legend()
plt.show()
