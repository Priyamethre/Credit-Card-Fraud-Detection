# CREDIT CARD FRAUD DETECTION

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
from sqlalchemy import create_engine
from urllib.parse import quote

# Importing  dataset using pandas
df = pd.read_csv("creditcard.csv")

df.info()

df.head()

df.columns

################ EDA

df.info()
df.describe()
df.shape

# Missing values check
df.isnull().sum()

# no missing or null values are found

#EDA using Autoviz

import sweetviz as sv
sweet_report = sv.analyze(df)

#Saving results to HTML file
sweet_report.show_html('sweet_report.html')

# Checking the distribution of the classes

classes = df['Class'].value_counts()
classes

normal_share = round((classes[0]/df['Class'].count()*100),2)
normal_share

fraud_share = round((classes[1]/df['Class'].count()*100),2)
fraud_share

# We can see that there is only 0.17% frauds.


#  DATA EXPLORATION

# Class Distribution (Fraud vs. Non-Fraud)
# Goal: Show the imbalance between fraudulent and non-fraudulent transactions.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['blue', 'red'])
plt.title("Fraud vs. Non-Fraud Transactions")
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()

# Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.


# TRANSACTION AMOUNT

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)
plt.show();


# Features correlation

plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()

# As expected, there is no notable correlation between features V1-V28. 
# There are certain correlations between some of these features and Time 
# (inverse correlation with V3) and Amount (direct correlation with V7 and V20, 
# inverse correlation with V1 and V5).

# Transaction Amount Distribution
# Goal: Show how fraud transactions differ in amount from legitimate ones.

plt.figure(figsize=(8, 5))
sns.kdeplot(df[df.Class == 0]['Amount'], label="Non-Fraud", shade=True, color='blue')
sns.kdeplot(df[df.Class == 1]['Amount'], label="Fraud", shade=True, color='red')
plt.title("Transaction Amount Distribution (Fraud vs. Non-Fraud)")
plt.legend()
plt.show()

# There are very few high-value transactions, making it a right-skewed distribution.
# The red (fraudulent transactions) and blue (non-fraudulent transactions) overlap significantly in the low amount range.
# This suggests that many fraud transactions occur at lower amounts, making it difficult to distinguish fraud based on amount alone.
# Some transactions extend beyond $25,000, but they are rare.

# Time vs. Fraudulent Transactions
#  Goal: Identify fraud patterns based on transaction time of day.


plt.figure(figsize=(8, 5))
sns.histplot(df[df.Class == 1]['Time'], bins=50, color='red', kde=True)
plt.title("Fraudulent Transactions Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Count")
plt.show()

# Fraudulent transactions are distributed across different time periods, meaning there is no single time where all fraud occurs.
# There are spikes around 50,000 seconds and 100,000 seconds, indicating that fraud happens more frequently at specific times.
# This suggests that fraudsters might be targeting transactions at particular hours when monitoring is lower.
# The red trend line shows that fraud transactions rise and fall over time, meaning fraud might follow a pattern based on transaction behavior.


# Box Plot of Transaction Amounts
#  Goal: See outliers in transaction amounts for fraud vs. non-fraud.

plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='Amount', data=df, palette=['blue', 'red'])
plt.title("Transaction Amounts for Fraud vs. Non-Fraud")
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()

# The fraud transactions (right box) show that most fraudulent transactions have low values.
# The range of fraud transactions is much smaller compared to non-fraud transactions
# The non-fraud transactions (left box) contain many high-value outliers, indicating that legitimate transactions often involve larger amounts.
# Fraudulent transactions rarely exceed a few thousand dollars, suggesting that fraudsters often operate with smaller amounts to avoid detection.
# This insight can be used to detect anomalies—high-value fraud cases may require further investigation.



# PCA (Principal Component Analysis) Scatter Plot
#  Goal: Reduce dimensions & visualize fraud vs. non-fraud transactions in 2D

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.drop(columns=['Class']))

plt.figure(figsize=(8, 5))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Class'], cmap='coolwarm', alpha=0.6)
plt.title("PCA Projection of Transactions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Fraud (1) vs Non-Fraud (0)")
plt.show()

# The majority of transactions (blue dots) are non-fraudulent (Class 0) and densely clustered along the lower region of the plot.
# Fraudulent transactions (red dots) are sparsely scattered, indicating their rarity.
# Fraudulent points do not cluster in one specific region but are spread across different PCA components.
# This suggests that fraud cannot be easily separated in a linear space, reinforcing the need for non-linear models (e.g., tree-based models or deep learning).


# Feature Distribution for Fraud vs. Non-Fraud
#  Goal: Compare key feature distributions for fraudulent vs. non-fraudulent transactions.

plt.figure(figsize=(10, 5))
sns.violinplot(x='Class', y='V1', data=df, palette=['blue', 'red'])
plt.title("Feature Distribution (V1) for Fraud vs. Non-Fraud")
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()

# The non-fraud transactions (blue) are mostly concentrated around 0, but there are some extreme negative values, indicating skewness in distribution.
# The fraudulent transactions (red) are more spread out and have a wider range, especially towards negative values.
# The fraud class distribution is more spread out, suggesting that Feature V1 has a different impact on fraud cases compared to non-fraud cases.
# This could mean Feature V1 is a strong predictor for distinguishing fraud.
# The long tail in non-fraud transactions suggests some extreme values in legitimate transactions, which might require further investigation.


# Outliers treatment
# We are not performing any outliers treatment for this particular dataset. Because all the columns are already PCA transformed, which assumed that the outlier values are taken care while transforming the data.


##### MODEL BUILDING

# pip install lightgbm
# pip install catboost
import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import sklearn
# pip install --upgrade xgboost imbalanced-learn


RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION

VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split

#CROSS-VALIDATION

NUMBER_KFOLDS = 5 #number of KFolds for cross-validation

RANDOM_STATE = 2018

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result

IS_LOCAL = False


target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
    
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )    
    

# Apply SMOTE only to train_df (Keep valid_df and test_df unchanged)
smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Increases fraud cases to 10%
X_train_resampled, y_train_resampled = smote.fit_resample(train_df[predictors], train_df[target])

#  Convert back to DataFrame (optional, for consistency with train_df format)
train_df_resampled = pd.DataFrame(X_train_resampled, columns=predictors)
train_df_resampled[target] = y_train_resampled  # Add the target column back



# RANDOM FOREST CLASSIFIER

clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

# Let's train the RandonForestClassifier using the train_df data and fit function.

clf.fit(train_df_resampled[predictors], train_df_resampled[target].values)

# Let's now predict the target values for the valid_df data, using predict function.
preds = clf.predict(valid_df[predictors])

# Let's also visualize the features importance.
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   

# The most important features are V14, V17, V16, V12, V10, V11.

# Confusion matrix

cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Area under curve

roc_auc_score(valid_df[target].values, preds)

# The ROC-AUC score obtained with RandomForrestClassifier is 0.87236.

#####  AdaBoostClassifier

clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                             n_estimators=NUM_ESTIMATORS)

clf.fit(train_df_resampled[predictors], train_df_resampled[target].values)

# Predict the target values
preds = clf.predict(valid_df[predictors])

# Features importance

tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 

# The most important features are V14,V4,V7,V12,Time and V8 

# Confusion matrix

cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# Area under curve
roc_auc_score(valid_df[target].values, preds)

# The ROC-AUC score obtained with AdaBoostClassifier is 0.87679.


#########  XGBoost

# Prepare the train and valid datasets
dtrain = xgb.DMatrix(train_df_resampled[predictors], train_df_resampled[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)

#What to monitor (in this case, **train** and **valid**)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE


model1 = xgb.train(params, 
                dtrain, 
                MAX_ROUNDS, 
                watchlist, 
                early_stopping_rounds=EARLY_STOP, 
                maximize=True, 
                verbose_eval=VERBOSE_EVAL)

# The best validation score (ROC-AUC) was 0.9984, for round 186.

# Plot variable importance

fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model1, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green") 
plt.show()

# The most important features are V14,V4,V17,V12, and V10

# Predict test set
# We used the train and validation sets for training and validation. We will use the trained model now to predict the target value for the test set.

preds = model1.predict(dtest)

roc_auc_score(test_df[target].values, preds)

# The AUC score for the prediction of fresh data (test set) is 0.9776.

### DECISION TREE MODEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize Decision Tree model
model2 = DecisionTreeClassifier(
    criterion="gini",  # or "entropy" for information gain
    max_depth=5,       # Limit tree depth to prevent overfitting
    min_samples_split=10,  # Minimum samples required to split a node
    min_samples_leaf=5,  # Minimum samples required in a leaf node
    random_state=RANDOM_STATE
)

# Train model on training data
model2.fit(train_df_resampled[predictors], train_df_resampled[target])

# Predict on validation data
valid_preds = model2.predict_proba(valid_df[predictors])[:, 1]

# Evaluate model performance
auc_valid = roc_auc_score(valid_df[target], valid_preds)
print(f"Validation ROC-AUC Score: {auc_valid:.4f}")
# Validation ROC-AUC Score: 0.9169

# Predict on test data
test_preds = model2.predict_proba(test_df[predictors])[:, 1]
auc_test = roc_auc_score(test_df[target], test_preds)
print(f"Test ROC-AUC Score: {auc_test:.4f}")
# Test ROC-AUC Score: 0.9369

# Feature Importance Plot
feature_importance = pd.Series(model2.feature_importances_, index=predictors).sort_values(ascending=False)

plt.figure(figsize=(8,5))
feature_importance.plot(kind="bar", color="green")
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# The most important features are V14,V17,V7,V12,Time and V10


# SAVING THE BEST MODEL

import joblib

# Save the trained model
joblib.dump(model1, "xgboost_fraud_model.pkl")



""" CONCLUSION
XGBoost shows a very high validation (0.984) and test AUC (0.977), meaning it generalizes well.
Decision Tree’s test AUC (0.9369) is slightly better than Random Forest (0.8723) but 
worse than XGBoost, indicating it may be overfitting slightly on the training data due to high variance.
Random Forest and AdaBoost show lower performance compared to XGBoost, meaning they 
might be underfitting compared to a more optimized model.

Final Verdict: XGBoost is the best model for fraud detection due to its superior generalization, high AUC, and ability to handle imbalanced data. 
"""

