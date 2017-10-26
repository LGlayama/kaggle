from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2, RFE


seed = 7
np.random.seed(seed)

def acc_cv(model):
    acc= cross_val_score(model, X_train, y, cv = 5)
    return(acc)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


train = pd.read_csv("train_final.csv",index_col=0)
test = pd.read_csv("test_final.csv", index_col=0)
y = train.Y


allfeatures= pd.concat((train.loc[:,'F1':'F27'],
                      test.loc[:,'F1':'F27']))

allfeatures = allfeatures.fillna(allfeatures.mean())

# plt.matshow(allfeatures.corr())
# plt.show()

# allfeatures=allfeatures.drop(['F23','F26'], 1)#correlation

# allfeatures=allfeatures.drop(['F6','F27'], 1)#high std

# allfeatures=allfeatures.drop(['F16','F17'], 1)# kBest
allfeatures=allfeatures.drop(['F3','F13'], 1)
# numeric_feats = allfeatures.dtypes[allfeatures.dtypes != "object"].index
# skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.95]
# skewed_feats = skewed_feats.index

allfeatures[list(allfeatures.columns.values)] = np.log1p(allfeatures[list(allfeatures.columns.values)])

# print (allfeatures.shape)

# allfeatures=Kbest_selector(allfeatures)

# print (allfeatures.shape)
# print (list(allfeatures.columns.values))



from sklearn.decomposition import PCA
pca1 = PCA(n_components=1)
pca1.fit(allfeatures[['F18','F26']])
newf=pca1.transform(allfeatures[['F18','F26']])

newfd=pd.DataFrame(data=newf[0:],index=range(1,99999),columns=['NF1'])  

allfeatures=allfeatures.drop(['F18','F26'], 1)

allfeatures = allfeatures.join(newfd)

# pca2 = PCA(n_components=1)
# pca2.fit(allfeatures[['F2','F14','F25']])
# newf2=pca2.transform(allfeatures[['F2','F14','F25']])

# newfd2=pd.DataFrame(data=newf2[0:],index=range(1,99999),columns=['NF2'])  

# allfeatures = allfeatures.join(newfd2)

dv=pd.get_dummies(allfeatures[['F2','F14','F25']])
allfeatures=allfeatures.drop(['F2','F14','F25'], 1)
allfeatures = allfeatures.join(dv)



train = allfeatures[:train.shape[0]]
test = allfeatures[train.shape[0]:]

# Outliers_to_drop = detect_outliers(train,3,list(train.columns.values))
# print(len(Outliers_to_drop))
# train= train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
# y= y.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# train.boxplot()
# plt.show()

X_train = train
X_test = test

from keras.models import Sequential
from keras.layers import Dense, Dropout



model = Sequential()
model.add(Dense(64, input_dim=24, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train= np.array(X_train)
y= np.array(y)
X_test = np.array(X_test)  
model.fit(X_train, y,
          epochs=20,
          batch_size=128,)
result = model.predict_proba(X_test, batch_size=32, verbose=0)
print (result.shape)
print (type(result))
solution = pd.DataFrame({"Id":range(49999,99999),"Y":result[:,0]}) 
solution.to_csv("keras.csv", index = False)

