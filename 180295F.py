# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-`1qonly "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os


import lightgbm
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, accuracy_score

import shap
import itertools

train = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/train.parquet")
test = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet")
train_labels = pd.read_csv("../input/amex-default-prediction/train_labels.csv")

"""# Describe"""


train.duplicated().sum()

for col in train.columns.values:
  print(col,'-',train[col].isna().sum()/len(train[col]))

non_numeric_cols = train.columns[train.dtypes == 'object'].values
non_numeric_cols

numeric_cols = train.columns[train.dtypes != 'object'].values
numeric_cols

import seaborn as sns
for i in numeric_cols:
    X = train[i].round(decimals = 3)
    plt.figure(i)
    ax = sns.countplot(X)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=5)
    plt.tight_layout()
    plt.show()

"""# Preprocessing"""

import matplotlib.style as style

style.use('seaborn-poster')
sns.set_style('ticks')
plt.subplots(figsize = (270,200))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(train.corr(), cmap=plt.get_cmap('Blues'), annot=True, mask=mask, center = 0, square=True,);
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 25);

"""# Preprocessing Train"""



train['Date'] =  pd.to_datetime(train['S_2'], format="%Y/%m/%d")
train['weekday'] = train['Date'].dt.weekday
train['day'] = train['Date'].dt.day
train['month'] = train['Date'].dt.month
train['year'] = train['Date'].dt.year

train['S_2'] = pd.to_numeric(train['S_2'].str.replace('-',''))
train['S_2']

train.drop(['Date'], axis=1, inplace=True)


cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",'customer_ID']

features = [col for col in train.columns.values if col not in cat_features]
features.append('customer_ID')

for i in cat_features:
    if train[i].dtype == 'int64':
        train.astype('int16')

train_cat = train[cat_features].groupby('customer_ID',as_index=False).agg(['count', 'last', 'nunique'])
train_cat.shape

for i in train.columns:
    if train[i].dtype == 'float64':
        train.astype('float16')



drop_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]

train.drop(drop_features, axis=1, inplace=True)

train = train.groupby('customer_ID',as_index=False).agg(['mean', 'std','sum','last'])



import gc
gc.collect()

train.columns.values

train = train.merge(train_cat, how='inner', on="customer_ID")


del train_cat

join_col = []
for i in train.columns.values:
    if type(i) is tuple:
        col = '_'.join(i)
        join_col.append(col)
train.columns = join_col
train.reset_index()

train = train.merge(train_labels, how='inner', on="customer_ID")


del train_labels

corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.98)]

# Drop features 

print(to_drop)



train.drop(to_drop, axis=1, inplace=True)


"""# Preprocessing Test"""



test['Date'] =  pd.to_datetime(test['S_2'], format="%Y/%m/%d")
test['weekday'] = test['Date'].dt.weekday
test['day'] = test['Date'].dt.day
test['month'] = test['Date'].dt.month
test['year'] = test['Date'].dt.year

test['S_2'] = pd.to_numeric(test['S_2'].str.replace('-',''))
test['S_2']

test.drop(['Date'], axis=1, inplace=True)


cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68",'customer_ID']

for i in test.columns.values:
    if test[i].dtype == 'int64' and i == 'customer_ID':
        test.astype('int16')



test_copy = test[cat_features].groupby('customer_ID',as_index=False).agg(['count', 'last', 'nunique'])

for i in test.columns.values:
    if test[i].dtype == 'float64':
        test.astype('float16')

drp_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

test.drop(drp_col, axis=1, inplace=True)

import gc
gc.collect()

test.head()

test = test.groupby('customer_ID',as_index=False).agg(['mean', 'std','sum',  'last'])




test = test.merge(test_copy, how='inner', on="customer_ID")

join_col = []
for i in test.columns.values:
    if type(i) is tuple:
        col = '_'.join(i)
        join_col.append(col)
        
test.columns = join_col
test.reset_index()

test.drop(to_drop, axis=1, inplace=True)


del test_copy

gc.collect()

"""# Training"""

target = train['target']
Features = train.drop('target', axis=1, inplace=False), 

numeric_cols = Features.columns[Features.dtypes != "object"].values
non_numeric_cols = Features.columns[Features.dtypes == 'object'].values

#test_1 = test.loc[:, Features.columns]


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

numeric_preprocessing_steps = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ])

non_numeric_preprocessing_steps = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #('onehot', OrdinalEncoder())
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


preprocessor = ColumnTransformer(
    transformers = [
        ("numeric", numeric_preprocessing_steps, numeric_cols),
        #("non_numeric",non_numeric_preprocessing_steps,non_numeric_cols)
    ],
    remainder='drop'
)

XGB = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)

from catboost import CatBoostClassifier

cat = CatBoostClassifier(iterations=3000, random_state=22,
                         learning_rate=0.03,
                         max_depth=9,
                         objective='Logloss',
                         subsample = 0.4,
                         colsample_bylevel=0.3
                        )

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(

    n_estimators= 3000, 
    num_leaves= 100,
    learning_rate= 0.01,
    colsample_bytree= 0.6,
    objective = 'binary',
    max_depth= 8,
    min_child_samples= 548,
    min_data_in_leaf = 27,
    bagging_freq = 7,
    bagging_fraction= 0.8,
    feature_fraction = 0.4,
    subsample = 0.8
)



full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", lgbm),
])

full_pipeline.fit(Features,target)

#import numpy as np
#np.mean(cross_val_score(full_pipeline, Features, train['target'], scoring='accuracy', cv=5))





import gc
gc.collect()

import joblib

joblib.dump(full_pipeline, 'pipe.joblib')

import joblib
full_pipeline = joblib.load('pipe.joblib')

test_probas = full_pipeline.predict_proba(test)

tests = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet")

tests = tests.groupby('customer_ID').tail(1)
print(tests.shape)

test_probas[:,1]

tests['prediction']=test_probas[:,1]
tests['prediction']

print(tests.shape)

sub = tests[['customer_ID','prediction']]
sub.shape

sub.to_csv("my_submission.csv", index=False)
sub.head()



"""# **Testing score**"""

from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(  
    Features,
    train['target'],
    test_size=0.2,
    shuffle=True,
    random_state=8
)

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

"""# **KNN**"""

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(15)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", KNN),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

"""# SVM"""

from sklearn.svm import SVC

svm = SVC(kernel='linear')

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", svm),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

"""# Light BGM"""
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators= 3000, 
    num_leaves= 100,
    learning_rate= 0.01,
    colsample_bytree= 0.6,
    objective = 'binary',
    max_depth= 8,
    min_data_in_leaf = 27,
    bagging_freq = 7,
    bagging_fraction= 0.8,
    feature_fraction = 0.4,
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", lgbm),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

"""# Output"""
full_pipeline.fit(Features,target)

joblib.dump(full_pipeline, 'pipes.joblib')

test_probas = full_pipeline.predict_proba(test)

test = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet")

tests = test.groupby('customer_ID').tail(1)

print(tests.shape)

tests['prediction']=test_probas[:,1]
print(tests['prediction'])

sub = tests[['customer_ID','prediction']]

sub.to_csv("my_submission.csv", index=False)


