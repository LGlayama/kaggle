import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoLarsCV

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

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


#part1

alphas1 = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]


cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas1]

cv_ridge = pd.Series(cv_ridge, index = alphas1)

print (cv_ridge)

alphas2 = [10,1, 0.1, 0.001, 0.0005,0.0001]

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas2]
cv_lasso=pd.Series(cv_lasso, index = alphas2)

print (cv_lasso)


sum_cof=[np.sum(Lasso(alpha = alpha).fit(X_train,y).coef_ !=0) for alpha in alphas2]

cv_ridge = pd.Series(sum_cof, index = np.log10(alphas2))
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("sum")

plt.show()


