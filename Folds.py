
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

import graphviz
import os
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random


""" 
#Reading the files and calculating the wheelsping % , stopout % for each student , grouped by assignment
df = pd.read_csv('skill_builders_16_17_all_features.csv')
print(df.columns)

df=df[['user_id','assignment_id','assignment_wheel_spin', 'assignment_stopout','next_assignment_wheel_spin', 'next_assignment_stopout']]
print(len(df.index))
df=df.drop_duplicates()
print(len(df.index))
#df=df.dropna()
print(len(df.index))

#df['ws_perc']=df.groupby(['user_id'])['assignment_wheel_spin'].sum()
#df['ass_sum'] = df['assignment_id'].groupby(df['user_id']).transform('count')
#df['ws_sum'] = df['assignment_wheel_spin'].groupby(df['user_id']).transform('sum')
df['ws_perc'] = df['assignment_wheel_spin'].groupby(df['user_id']).transform('sum') / df['assignment_id'].groupby(df['user_id']).transform('count')
df['so_perc'] = df['assignment_stopout'].groupby(df['user_id']).transform('sum') / df['assignment_id'].groupby(df['user_id']).transform('count')
df['nws_perc'] = df['next_assignment_wheel_spin'].groupby(df['user_id']).transform('sum') / (df['assignment_id'].groupby(df['user_id']).transform('count')-1)
df['nso_perc'] = df['next_assignment_stopout'].groupby(df['user_id']).transform('sum') / (df['assignment_id'].groupby(df['user_id']).transform('count')-1)
#df['numberofassgn']=df.groupby(['user_id'])['assignment_id'].count()
#print(df[df['user_id']==126101])

df=df[['user_id','ws_perc','so_perc','nws_perc','nso_perc']]
df=df.drop_duplicates()

df.to_csv('fold_data.csv')
"""

df=pd.read_csv("fold_data.csv")
df=df.fillna(0)
print(len(df.index))
# Getting the values and plotting it
f1 = df['ws_perc'].values
f2 = df['so_perc'].values
f3 =df['nws_perc'].values
f4= df['nso_perc'].values

X = np.array(list(zip(f1, f2,f3,f4)))
#plt.scatter(f1, f2,f3, c='black', s=7)
#plt.show()
X = np.array(list(zip(f1,f2))).reshape(len(f1), 2)
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeanModel = KMeans(n_clusters=10).fit(X)
kmeanModel.fit(X)
y=kmeanModel.predict(X)
print(np.unique(y))
df['Cluster']=y
print(df[0:10])


print(df.groupby(['Cluster'])['user_id'].count())

Cluster0=list(df.loc[df['Cluster']==0]['user_id'])
Cluster1=list(df.loc[df['Cluster']==1]['user_id'])
Cluster2=list(df.loc[df['Cluster']==2]['user_id'])
Cluster3=list(df.loc[df['Cluster']==3]['user_id'])
Cluster4=list(df.loc[df['Cluster']==4]['user_id'])
Cluster5=list(df.loc[df['Cluster']==5]['user_id'])
Cluster6=list(df.loc[df['Cluster']==6]['user_id'])
Cluster7=list(df.loc[df['Cluster']==7]['user_id'])
Cluster8=list(df.loc[df['Cluster']==8]['user_id'])
Cluster9=list(df.loc[df['Cluster']==9]['user_id'])



#print(Cluster0)
#Shuffling the Cluster lists so that there is some amount of randomness in assigning the folds and not all
# userids in the ascending order get assigned to the first folds
Cluster0=random.sample(Cluster0, len(Cluster0))
Cluster1=random.sample(Cluster1, len(Cluster1))
Cluster2=random.sample(Cluster2, len(Cluster2))
Cluster3=random.sample(Cluster3, len(Cluster3))
Cluster4=random.sample(Cluster4, len(Cluster4))
Cluster5=random.sample(Cluster5, len(Cluster5))
Cluster6=random.sample(Cluster6, len(Cluster6))
Cluster7=random.sample(Cluster7, len(Cluster7))
Cluster8=random.sample(Cluster8, len(Cluster8))
Cluster9=random.sample(Cluster9, len(Cluster9))
#print(Cluster0)
#Number of folds
k=10
#Trying to assign students in each cluster to a folds

df['fold']=-1

for i in range(0,len(Cluster0)):

    #df.loc[df['user_id'] == Cluster0[i]]['fold'] = i % k
    df.loc[df['user_id'] == Cluster0[i], 'fold'] = i%k
print("Done cluster ")


for i in range(0,len(Cluster1)):
    df.loc[df['user_id'] == Cluster1[i], 'fold'] = i % k
print("Done cluster ")

for i in range(0,len(Cluster2)):
    df.loc[df['user_id'] == Cluster2[i], 'fold'] = i % k
print("Done cluster ")

for i in range(0,len(Cluster3)):
    df.loc[df['user_id'] == Cluster3[i], 'fold'] = i % k
print("Done cluster ")
for i in range(0,len(Cluster4)):
    df.loc[df['user_id'] == Cluster4[i], 'fold'] = i % k

print("Done cluster ")
for i in range(0,len(Cluster5)):
    df.loc[df['user_id'] == Cluster5[i], 'fold'] = i % k


for i in range(0,len(Cluster6)):
    df.loc[df['user_id'] == Cluster6[i], 'fold'] = i % k
print("Done cluster ")

for i in range(0,len(Cluster7)):
    df.loc[df['user_id'] == Cluster7[i], 'fold'] = i % k
print("Done cluster ")
for i in range(0,len(Cluster8)):
    df.loc[df['user_id'] == Cluster8[i], 'fold'] = i % k
print("Done cluster ")
for i in range(0,len(Cluster9)):
    df.loc[df['user_id'] == Cluster9[i], 'fold'] = i % k
print("Done cluster ")


print(df[0:5])

df.to_csv("Folded_data_4.csv")
df_fold=df[['user_id','fold']]
df_fold.to_csv('User_id_folds.csv')
print(set(list(df_fold['fold'])))

print(df_fold.groupby(['fold'])['user_id'].count())

us=set(list(df_fold['user_id']))
print(len(df_fold.index))
print(len(us))