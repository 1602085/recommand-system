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
theta1 = np.random.randint(1, 6, size=(b, q)) # for find similar mtr
theta2 = np.zeros([b,q])
x = np.random.randn(p, b)

alpha = 0.000015  # regularization parameter very small because sum of theta1 very big
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

def gradientDescent(a, theta2, alpha, beta, num_iter):
    u, s, v = np.linalg.svd(a)
    s1 = np.sqrt(s)
    s2 = len(u)
    s3 = np.zeros([s1 s2])
    
    for f in range(len(s1)):
        s3[f-1][f-1] = s1[f-1]
    s3 = s3[:b, :b]
    s4 = s2 - len(s3)
    s5 = np.zeros([s4, s4])
    s6 = np.concatenate([s3, s5])
    theta3 = np.matmul(u, s6)
    
    s7 = len(v) - len(s3)
    s8 = np.zeros([s7,s7])
    s9 = np.concatenate([s3, s8], axis=1)
    theta4 = np.matmul(s9, v)
    
    for k in range(num_iter):
        for i in range(theta3.shape[0]):
            theta2 = theta2 + theta4*np.transpose(theta3[i, :])
            for l in range(theta4.shape[1]):
                theta3[i,:] = theta3[i,:] - beta(np.sum(theta3[i,:]*theta2[:,l]-a[i][l])*np.transpose(theta2[:,l]) + alpha*np.sum(theta3[i,:]))
                theta4[:,l] = theta4[:,l] - beta(np.sum(theta3[i,:]*theta2[:,l]-a[i][l])*np.transpose(theta3[i,:]) + alpha*np.sum(theta4[:,l]))
        cost.append(CostFunction(a, theta3, theta4, alpha))
    return cost, theta3, theta4
                
cost, theta3, theta4  = gradientDescent(a, theta2, alpha, beta, num_iter)

