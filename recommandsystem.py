import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer # for Convert text to matrix of token counts 


df = pd.read_csv("/home/shikha/Desktop/rating.csv")
df.columns = ['userId', 'MovieId', 'rating', 'timestamp']

df1 = pd.read_csv('/home/shikha/Desktop/movies5.csv')
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

r = ['scifi', 'filmnoir'] # so again append scifi and filmnoir

m.extend(r)
print(m)  # finally get all type of movie genres(unique)


p = max(df.userId)
q = max(df.MovieId)
a = np.zeros([p, q])
b = len(m)
theta1 = np.random.randint(1, 6, size=(p, b)) # for find similar matr
#theta1 = np.random.random(p*b).reshape(p,b)
theta2 = np.random.random(b*q).reshape(b,q)
theta22 = np.transpose(theta2)
userId_unique = df.userId.unique()
print(theta1.shape)
print(theta2.shape)

beta = 0.004      # for learning optimization
num_iter = 30   


for i in userId_unique:
    df3 = df[df['userId'] == i].iloc[:]
    df3.reset_index(inplace=True)
    for j in range(len(df3)):
        n = df3.MovieId[j]
        a[i-1][n-1] = df.rating[j]   # get ratings of userId-movieId, size (p*q) matrix

print(a.shape)


def costFunction(a, theta1, theta22):
    for m in range(len(a)):
        for u in range(len(a[0])):
            error = (a[m][u] - np.tranpose(a[m][u] - np.matmul(theta1[m], theta22[u])))*2
    return error


output1 = []
output2 = []

def SochasticGredientDescent(a, theta1, theta22, theta2, beta):
    for i in range(num_iter):
        for m in range(len(theta1)):
            for k in range(len(theta1[0])):
                for u in range(len(theta2[0])):
                    theta1[m][k] = theta1[m][k] + 2*beta*(a[m][u] - np.matmul(theta1[m], theta22[u]))*theta2[k][u]
        for k in range(len(theta2)):
            for u in range(len(theta2[0])):
                for m in range(len(theta1)):
                    theta2[k][u] = theta2[k][u] - 2*beta*(a[m][u] - np.matmul(theta1[m], theta22[u]))*theta1[m][k]  
        output1.append(theta1)
        output2.append(theta2)
    return output1[-1], output2[-1]

cost1 = SochasticGredientDescent(a, theta1, theta22, theta2, beta)

user_genres = cost1[0]
genres_movieId = cost1[1]


user_movidId_rating = np.matmul(user_genres, genres_movieId)  #finally get how much rating, which user give which movie
print(user_movieId_rating)
