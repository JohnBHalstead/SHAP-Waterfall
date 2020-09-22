# SHAP-Waterfall

**Install**

Using pip (recommended)
    
    pip install shap-waterfall
    
**Introduction**

Many times when VMware Data Science Teams present their Machine Learning models' propensity to buy scores (estimated probabilities) to stakeholders, stakeholders ask why a customer's propensity to buy is higher than the other customer. The stakeholder's question was our primary motivation. We were further motivated by recent algorithm transparency language in the EU's General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA). Although the "right to explanation" is not necessarily clear, our desire is to act in good faith by providing local explainability between two observations, clients, and customers.

This graph solution provides a local classification model interruptibility between two observations, which we call customers. It uses each customer's estimated probability and fills the gap between the two probabilities with SHAP values that are ordered from higher to lower importance. We prefer SHAP over others (for example, LIME) because of its concrete theory and ability to fairly distribute effects.

Currently, this package only works for tree and tree ensemble classification models. Our decision to limit the use to tree methods was based on two considerations. We desired to take advantage of the tree explainer's speed. As a business practice, we tend to deploy Random Forest, XGBoost, LightGBM, and  other tree ensembles more often than other classifications methods.

However, we plan to include the kernel explainer in future versions.

The package requires a tree classifier, training data, validation/test/scoring data with a column titled "Customer", the two observations of interest, and the desired number of important features. The package produces a Waterfall Chart. 

ShapWaterFall(*clf, X_tng, X_val, observation1, observation2, num_features*)

**Required**

*clf*: a tree based classifier that is fitted to X_tng, training data.

*X_tng*: the training Data Frame used to fit the model.

*X_val*: the validation, test, or scoring Data Frame under observation. Note that the data frame must contain an extra column who's label is "Customer".

*observation1 and observation2*: the first observation, client, or customer under study. If the column data is a string, use "observation1". Otherwise, use an integer, i.e., 4 or 107, etc. 

*num_features*: the number of important features that describe the local interpretability between to the two observations. 

**Examples**

**packages**

import pandas as pd

import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, RandomizedSearchCV

import shap-waterfall

**models**

rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)

**load and organize Wisconsin Breast Cancer Data**

data = load_breast_cancer()

label_names = data['target_names']

labels = data['target']

feature_names = data['feature_names']

features = data['data']

**data splits**

X_tng, X_val, y_tng, y_val = train_test_split(features, labels, test_size=0.33, random_state=42)

print(X_tng.shape) # (381, 30)

print(X_val.shape) # (188, 30)

X_tng = pd.DataFrame(X_tng)

X_tng.columns = feature_names

X_val = pd.DataFrame(X_val)

X_val.columns = feature_names

**fit classifiers and measure AUC**

clf = rf_clf.fit(X_tng, y_tng)

pred_rf = clf.predict_proba(X_val)

score_rf = roc_auc_score(y_val,pred_rf[:,1])

print(score_rf, 'Random Forest AUC')

*0.9951893425434809 Random Forest AUC*

**IMPORTANT: add a 'Customer' column to the val/test/score data**

X_val = pd.DataFrame(X_val)

X_val['Customer'] = X_val.index

print(X_val.shape) # (188, 31)

**Use Case 1**

ShapWaterFall(clf, X_tng, X_val, 5, 100, 5)

ShapWaterFall(clf, X_tng, X_val, 100, 5, 7)

**Use Case 2**

ShapWaterFall(clf, X_tng, X_val, 36, 94, 5)

ShapWaterFall(clf, X_tng, X_val, 94, 36, 7)

**Authors**

John Halstead, jhalstead@vmware.com

Ravi Prasad K, rkondapalli@vmware.com

Rajesh Vikraman, rvikraman@vmware.com

Kiran R, rki@vmware.com


 
