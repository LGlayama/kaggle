import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, Lasso, LassoLarsCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train_t = all_data[:train.shape[0]]
X_test_t = all_data[train.shape[0]:]

y_t = train.SalePrice

ntrain =X_train_t.shape[0]
ntest = X_test_t.shape[0]
kf=KFold(n_splits=5,random_state=2017)


def stacking1(clf,X_train,y_train,X_test):
	oof_train=np.zeros((ntrain,))
	oof_test=np.zeros((ntest,))
	oof_test_skf=np.empty((5,ntest))

	for i, (train_index,test_index) in enumerate(kf.split(X_train)):
		kf_X_train=X_train.iloc[train_index]
		kf_y_train=y_train.iloc[train_index]
		kf_X_test= X_train.iloc[test_index]

		clf.fit(kf_X_train,kf_y_train)

		oof_train[test_index]=clf.predict(kf_X_test)
		oof_test_skf[i:]=clf.predict(X_test)

	oof_test[:]=oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


las = Lasso(alpha = 0.0005)

ela = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000)

rid=Ridge(alpha=10)


F1,T1= stacking1(las,X_train_t,y_t,X_test_t)
F2,T2= stacking1(ela,X_train_t,y_t,X_test_t)
F3,T3= stacking1(rid,X_train_t,y_t,X_test_t)

NF=np.column_stack((X_train_t,F1,F2,F3))
NT=np.column_stack((X_test_t,T1,T2,T3))

Layer_2_ridge=Ridge(alpha=10)

Layer_2_ridge.fit(NF,y_t)

Layer_2_y=Layer_2_ridge.predict(NT)
Layer_2_y=np.expm1(Layer_2_y)

solution = pd.DataFrame({"Id":test.Id,"SalePrice":Layer_2_y}) 

solution.to_csv("las+ela+rid+all.csv", index = False)
