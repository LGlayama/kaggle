from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2, RFE
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

def stacking1(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

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

allfeatures = allfeatures.fillna(allfeatures.mean())

# plt.matshow(allfeatures.corr())
# plt.show()

# allfeatures=allfeatures.drop(['F23','F26'], 1)#correlation

# allfeatures=allfeatures.drop(['F6','F27'], 1)#high std

# allfeatures=allfeatures.drop(['F16','F17'], 1)# kBest
allfeatures=allfeatures.drop(['F13','F3'], 1)
# numeric_feats = allfeatures.dtypes[allfeatures.dtypes != "object"].index
# skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.95]
# skewed_feats = skewed_feats.index

allfeatures[list(allfeatures.columns.values)] = np.log1p(allfeatures[list(allfeatures.columns.values)])
# scaler = StandardScaler()
# allfeatures[list(allfeatures.columns.values)] =scaler.fit_transform(allfeatures[list(allfeatures.columns.values)])
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
# pca2.fit(allfeatures[['F3','F23']])

# newf2=pca2.transform(allfeatures[['F3','F23']])

# allfeatures=allfeatures.drop(['F3'], 1)

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
ntrain =X_train.shape[0]
ntest = X_test.shape[0]
kf=KFold(ntrain, n_folds= 5, random_state=0)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV



# rf = RandomForestClassifier()

# param_grid1 = {
#                  'n_estimators': [170,180,190],
#                  'max_depth': [11,13,15,17,19]
#              }

# grid_clf1 = GridSearchCV(rf, param_grid1, cv=5)
# grid_clf1.fit(X_train, y)

# print (gsearch1.grid_scores_)
# print (gsearch1.best_params_)
# print (gsearch1.best_score_)

# param_grid2 = {
#                  'n_neighbors': [3,5,7,9],
#              }

# knn = KNeighborsClassifier()

# grid_clf2 = GridSearchCV(knn, param_grid2, cv=5)
# grid_clf2.fit(X_train, y)

# print (grid_clf2. best_params_)

# param_grid3 = {
#                  'n_estimators': [150,200,250,300],
#                  'learning_rate' : [0.1,0.5,1.0]
#              }
# ada=AdaBoostClassifier()

# grid_clf3 = GridSearchCV(ada, param_grid3, cv=5)
# grid_clf3.fit(X_train, y)

# print (grid_clf3. best_params_)


# param_grid4 = {
#                   'n_estimators': [150,180,200],
#                   'max_depth': [9,11,13,15]
#              }
# ef=ExtraTreesClassifier()

# grid_clf4 = GridSearchCV(ef, param_grid4, cv=5)
# grid_clf4.fit(X_train, y)

# print (grid_clf4. best_params_)

# param_grid5 = {
#                   'n_estimators': [150,180,200],
#                   'max_depth': [9,11,13,15]
#              }

# gb=GradientBoostingClassifier()

# grid_clf5 = GridSearchCV(gb, param_grid5, cv=5)
# grid_clf5.fit(X_train, y)

# print (grid_clf5. best_params_)

#rf = RandomForestClassifier(n_estimators=180,max_depth=11)
# knn=KNeighborsClassifier(n_neighbors=9)
# ada=AdaBoostClassifier(n_estimators = 250, learning_rate= 0.1)
# ef=ExtraTreesClassifier(n_estimators = 200, max_depth = 15)

#rf.fit(X_train,y)
# print (acc_cv(rf).mean())
# eclf1 = VotingClassifier(estimators=[('rf', rf), ('knn', knn), ('ada', ada),('ef',ef)], voting='soft')

# eclf1.fit(X_train,y)
# print (acc_cv(eclf1).mean())
# y_p=eclf1.predict_proba(X_test)

# parameters = {'n_estimators': [1,5,10,20,30],}
# eclf2=BaggingClassifier()

# grid_clf6=GridSearchCV(BaggingClassifier(rf),
#             			parameters,
#             			scoring="roc_auc").fit(X_train, y)
# print (grid_clf6. best_params_)

# bc=BaggingClassifier(rf,n_estimators=10)
# bc.fit(X_train, y)
# y_p=bc.predict_proba(X_test)
#F1,T1= stacking1(rf,X_train,y,X_test)
# F2,T2= stacking1(knn,X_train,y,X_test)
# F3,T3= stacking1(ada,X_train,y,X_test)
# F4,T4= stacking1(ef,X_train,y,X_test)


# NF=np.column_stack((X_train,F2,F3,F4))
# NT=np.column_stack((X_test,T2,T3,T4))


# import xgboost as xgb

# param_test1 = {

#  'max_depth':range(1,10,2),

#  'min_child_weight':[1,3,5];

# }



# gsearch1 = grid_search.GridSearchCV(

# estimator = xgb.XGBClassifier(

#                 learning_rate=0.2,

#                 n_estimators=200, 

#                 max_depth=4,

#                 min_child_weight=1.5,

#                 gamma=0.0,

# 				subsample=0.2,

#                 colsample_bytree=0.2,                                                                

#                 reg_alpha=0.9,

#                 seed=42,

#                 silent=1),

# 		param_grid = param_test1,

# 		n_jobs=4,

# 		iid=False,

# 		cv=5)

# gsearch1.fit(X_train,y)

# print (gsearch1.best_params_)

# rf= RandomForestClassifier(n_estimators=180,max_depth=11)

# print (acc_cv(random_forest))
# rfecv = RFE(estimator=random_forest, n_features_to_select=26,step=1)
# rfecv.fit(X_train, y)
# y_p=rfecv.predict_proba(X_test)


# rf.fit(X_train,y)
# y_p=rf.predict_proba(X_test)


# solution = pd.DataFrame({"Id":range(49999,99999),"Y":y_p[:,1]}) 

# solution.to_csv("rf_nk.csv", index = False)

