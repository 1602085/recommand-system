import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer # for Convert text to matrix of token counts 


df = pd.read_csv("/home/shikha/Downloads/rating.csv")
df.columns = ['userId', 'MovieId', 'rating', 'timestamp']

df1 = pd.read_csv('/home/shikha/movies5.csv')
df1.columns = ['movietype', 'movieID', 'one', 'two', 'three', 'four', 'five']  # there is four commas in csv so we give one, two, three, four name and later filter out movietype and movieId in this dataframe

df1 = df1.iloc[:, 0:2] # only movietype and movieID column filterout from dataframe

print(df1.head())

s=[]  # for collect different types of movie genres(all movies)

for i in range(len(df1)):
    z = df1.movietype[i].split('|')
    s.extend(z)
    
bow = CountVectorizer() 
x = bow.fit_transform(s) 
m = bow.get_feature_names() # get the genres name

n = ['sci', 'fi', 'film', 'noir'] # there if hyphen between sci-fi and film-noir so they consider different genres 

for i in n:
    m.remove(i) 

r = ['scifi', 'filmnoir'] # so again append scifi and film-noir

m.extend(r)
print(m)  # finally get all type of movie genres(unique)


p = max(df.userId)
q = max(df.MovieId)
a = np.zeros([p, q])
b = len(m)
theta = np.random.randint(1, 6, size=(p, q)) # for find similar a
theta1 = np.zeros([p,q])
x = np.random.randn(q, b)

alpha = 0.000002  # regularization parameter very small because sum of theta1 very big
beta = 0.001      # for learning optimization
num_iter = 300    
cost = []


for i in range(1, p+1):
    for j in range(len(df.MovieId)): 
        if df.userId[j] == i:
           n = df.MovieId[j]
           a[i-1][n-1] = df.rating[j]   # get ratings of userId-movieId, size (p*q) matrix
          

def costFunction(a, theta1, x, alpha):
    j = 0.5*((np.sum(theta1-a)**2) + alpha*np.sum(x**2) + alpha*np.sum(theta1**2)) 
    return j

def gradientDescent(a, theta, x, alpha, beta, num_iter):
    for k in range(num_iter):
        for i in range(a.shape[0]):          # for all user
            for l in range(x.shape[1]):      # for different type of movie generes
                theta1 = theta1 + theta*np.transpose(x[:,l])  
     
                x[:,l] = x[:,l] - beta(np.sum(theta1-a)*np.transpose(theta1[i,:]) + alpha*np.sum(x[:,l]))  
                theta1[i,:] = theta1[i,:] - beta(np.sum(theta1-a)*np.transpose(x[:,l]) + alpha*np.sum(theta1[i,:]))
        cost.append(costFunction(a, theta1, x, alpha))           
    return cost, x, theta1
               

cost, x, theta, theta1  = gradientDescent(a, theta, x, alpha, beta, num_iter)

