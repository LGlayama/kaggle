from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score,KFold

np.random.seed(0)

def stacking1(clf,X_train,y_train,X_test):
	oof_train=np.zeros((ntrain,))
	oof_test=np.zeros((ntest,))
	oof_test_skf=np.empty((5,ntest))

	for i, (train_index,test_index) in enumerate(kf.split(X_train)):
		kf_X_train=X_train.iloc[train_index]
		kf_y_train=y_train.iloc[train_index]
		kf_X_test= X_train.iloc[test_index]

		clf.fit(kf_X_train,kf_y_train)

		oof_train[test_index]=clf.predict_proba(kf_X_test)[:,1]
		oof_test_skf[i:]=clf.predict_proba(X_test)[:,1]

	oof_test[:]=oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def stacking2(clf,X_train,y_train,X_test):
	oof_train=np.zeros((ntrain,))
	oof_test=np.zeros((ntest,))
	oof_test_skf=np.empty((5,ntest))

	for i, (train_index,test_index) in enumerate(kf.split(X_train)):
		kf_X_train=X_train.iloc[train_index]
		kf_y_train=y_train.iloc[train_index]
		kf_X_test= X_train.iloc[test_index]
		
		kf_X_train= np.array(kf_X_train)
		kf_y_train= np.array(kf_y_train)
		kf_X_test = np.array(kf_X_test)
		X_test=np.array(X_test)

		clf.fit(kf_X_train,kf_y_train)

		oof_train[test_index]=clf.predict_proba(kf_X_test)[:,0]
		oof_test_skf[i:]=clf.predict_proba(X_test)[:,0]

	oof_test[:]=oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def acc_cv(model):
    acc= cross_val_score(model, X_train, y, scoring="accuracy", cv = 5)
    return(acc)
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

allfeatures = allfeatures.fillna(allfeatures.mean())

# plt.matshow(allfeatures.corr())
# plt.show()

# allfeatures=allfeatures.drop(['F23','F26'], 1)#correlation

# allfeatures=allfeatures.drop(['F6','F27'], 1)#high std

allfeatures=allfeatures.drop(['F3','F13'], 1)
# numeric_feats = allfeatures.dtypes[allfeatures.dtypes != "object"].index
# skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.95]
# skewed_feats = skewed_feats.index

allfeatures[list(allfeatures.columns.values)] = np.log1p(allfeatures[list(allfeatures.columns.values)])
# scaler = StandardScaler()
# allfeatures[list(allfeatures.columns.values)] =scaler.fit_transform(allfeatures[list(allfeatures.columns.values)])
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(allfeatures[['F2','F14','F25']])
# newf=pca.transform(allfeatures[['F2','F14','F25']])

# newfd=pd.DataFrame(data=newf[0:,0:],index=range(1,99999),columns=['NF1','NF2'])  

# allfeatures=allfeatures.drop(['F2','F14','F25'], 1)

# allfeatures = allfeatures.join(newfd)
from sklearn.decomposition import PCA
pca1 = PCA(n_components=1)
pca1.fit(allfeatures[['F18','F26']])
newf=pca1.transform(allfeatures[['F18','F26']])

newfd=pd.DataFrame(data=newf[0:],index=range(1,99999),columns=['NF1'])  

allfeatures=allfeatures.drop(['F18','F26'], 1)

allfeatures = allfeatures.join(newfd)




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

ntrain =X_train.shape[0]
ntest = X_test.shape[0]
kf=KFold(n_splits=5,random_state=2017)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
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



knn=KNeighborsClassifier(n_neighbors=9)
ada=AdaBoostClassifier(n_estimators = 250, learning_rate= 0.1)
ef=ExtraTreesClassifier(n_estimators = 200, max_depth = 15)


F2,T2= stacking1(knn,X_train,y,X_test)
F3,T3= stacking1(ada,X_train,y,X_test)
F4,T4= stacking1(ef,X_train,y,X_test)
F5,T5= stacking2(model,X_train,y,X_test)

NF = np.concatenate((F2,F3,F4,F5), axis=1)
NT = np.concatenate((T2,T3,T4,T5), axis=1)

# rf = RandomForestClassifier()

# param_grid = {
#                  'n_estimators': [100,150,180,200,250],
#                  'max_depth': [  15,17,20,23,25]
#              }
# # knn = KNeighborsClassifier(n_neighbors = 3)

# grid_clf =  grid_search.GridSearchCV(rf, param_grid, cv=10)
# grid_clf.fit(NF, y)

# print (grid_clf. best_params_)
# print (grid_clf. best_score_)
# print (grid_clf. grid_scores_)


rf = RandomForestClassifier(n_estimators=200,max_depth=15)
rf.fit(NF,y)
y_p=rf.predict_proba(NT)



# # x.fit(NF,y)
# #y_p=x.predict_proba(NT)

solution = pd.DataFrame({"Id":range(49999,99999),"Y":y_p[:,1]}) 

solution.to_csv("xgb_rfk.csv", index = False)
