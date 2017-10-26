import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.model_selection import cross_val_score

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
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# from sklearn.tree import DecisionTreeRegressor

# clf = DecisionTreeRegressor(max_depth=8)

# rmse= np.sqrt(-cross_val_score(clf, X_train, y, scoring="neg_mean_squared_error", cv = 5))
# print(rmse)

# clf.fit(X_train, y) 

# y_p=clf.predict(X_test)
# y_p_t=[]

# for i in  y_p:
# 	y_p_t.append( np.expm1(i))

# solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p_t}) 

# solution.to_csv("dt_sol.csv", index = False)



# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV

# clf = SVR(C=1.0, epsilon=0.2,kernel='sigmoid')

# rmse= np.sqrt(-cross_val_score(clf, X_train, y, scoring="neg_mean_squared_error", cv = 5))
# print(rmse)

# clf.fit(X_train, y) 

# y_p=clf.predict(X_test)
# y_p_t=[]

# for i in  y_p:
# 	y_p_t.append( np.expm1(i))

# solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p_t}) 

# solution.to_csv("svr_sol.csv", index = False)



# from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score

# from sklearn.decomposition import PCA
# pca = PCA(n_components=200)
# pca.fit(X_train)

# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, pca.transform(X_train), y, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)
# print (rmse_cv(LassoCV(alphas = [1, 0.1, 0.001, 0.0005])).mean())

# clf = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

# clf.fit(X_train,y)

# y_p=clf.predict(X_test)
# y_p_t=[]

# for i in  y_p:
# 	y_p_t.append( np.expm1(i))

# solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p_t}) 

# solution.to_csv("lassoCV_sol.csv", index = False)



# from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.decomposition import PCA
# pca = PCA(n_components=200)
# pca.fit(X_train)

# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, pca.transform(X_train), y, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)

# clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
#                                                min_samples_leaf=15, min_samples_split=10, loss='huber')

# print (rmse_cv(clf).mean())

# clf.fit(X_train,y)

# y_p=clf.predict(X_test)
# y_p_t=[]

# for i in  y_p:
# 	y_p_t.append( np.expm1(i))

# solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p_t}) 

# solution.to_csv("gbr_sol.csv", index = False)


# clf = ElasticNet(random_state=2)

# print (rmse_cv(clf).mean())

# clf.fit(X_train,y)

# y_p=clf.predict(X_test)
# y_p_t=[]

# for i in  y_p:
# 	y_p_t.append( np.expm1(i))

# solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p_t}) 

# solution.to_csv("gbr_sol.csv", index = False)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor



clf1 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

clf1.fit(X_train,y)

y_p1=clf1.predict(X_test)
y_p1=np.expm1(y_p1)

clf2 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

clf2.fit(X_train,y)

y_p2=clf2.predict(X_test)
y_p2=np.expm1(y_p2)

clf3=AdaBoostRegressor(n_estimators=360, learning_rate=0.1) 
clf3.fit(X_train,y)

y_p3=clf3.predict(X_test)
y_p3=np.expm1(y_p3)

clf4=Ridge(alpha=10)
clf4.fit(X_train,y)

y_p4=clf4.predict(X_test)
y_p4=np.expm1(y_p4)

y_p=0.65*y_p2 + 0.35*y_p3

solution = pd.DataFrame({"Id":test.Id,"SalePrice":y_p}) 

solution.to_csv("mix_sol_6.csv", index = False)






