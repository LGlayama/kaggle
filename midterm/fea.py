from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2, RFE

def stack(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def acc_cv(model):
    acc= cross_val_score(model, X_train, y, cv = 5)
    return(acc)


def VarianceThreshold_selector(data):
    columns = data.columns
    selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
    selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True) if x]
    return pd.DataFrame(selector.fit_transform(data), columns=labels)

def Kbest_selector(data):
    columns = data.columns
    selector = SelectKBest(chi2, k=23)
    selector.fit_transform(train,y)
    labels = [columns[x] for x in selector.get_support(indices=True) if x]
    return pd.DataFrame(selector.fit_transform(data), columns=labels)

def detect_outliers(df,n,features):

    outlier_indices = []

    for col in features:
        
        Q1 = np.percentile(df[col], 25)       
        Q3 = np.percentile(df[col],75)      
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


train = pd.read_csv("train_final.csv",index_col=0)
test = pd.read_csv("test_final.csv", index_col=0)
y = train.Y


allfeatures= pd.concat((train.loc[:,'F1':'F27'],
                      test.loc[:,'F1':'F27']))



allfeatures['F19'] = allfeatures['F19'].fillna(allfeatures['F19'].mean())
allfeatures['F5'] = allfeatures['F5'].fillna(allfeatures['F5'].value_counts().index[0])


print (pd.isnull(allfeatures).sum() > 0)
print (allfeatures.shape)

# plt.matshow(allfeatures.corr())
# plt.show()

# allfeatures=allfeatures.drop(['F23','F26'], 1)#correlation

# allfeatures=allfeatures.drop(['F6','F27'], 1)#high std

# allfeatures=allfeatures.drop(['F16','F17'], 1)# kBest
# allfeatures=allfeatures.drop(['F3','F13'], 1)# kBest
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
# train_new=Kbest_selector(train)

# print (list(train_new.columns.values))

random_forest = RandomForestClassifier(n_estimators=180,max_depth=11)

rfecv = RFE(estimator=random_forest, n_features_to_select=24,step=1)
rfecv.fit(X_train,y)
print (list(X_train.columns.values))
print (rfecv.support_ )

