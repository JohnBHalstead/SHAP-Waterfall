# SHAP-Waterfall

**Install**

Using pip (recommended)
    
    pip install shap-waterfall
    
**Introduction**

Many times when VMware Data Science Teams present their ML models' propensity to buy scores (estimated probabilities) to stakeholders,  stakeholders ask why a customer’s propensity to buy is higher than the other customer. Their question was our motivation.  Plus recent EU and US mandates that require machine learning model explainability became another motivation.

This graph solution provides a local classification model interruptibility between two observations, which we call customers. It uses each customer's estimated probability and fills the gap between the two probabilities with SHAP values that are ordered from higher to lower importance.

Currently, this package only works for tree and tree ensemble classification models. Our decision to limit the use to tree methods was based on two considerations. We desired to take advantage of the tree explainer's speed. As a business practice, we tend to deploy Random Forest, XGBoost, LightGBM, and  other tree ensembles more often than other classifications methods.

However, we plan on including the kernel explainer in future versions.

The package requires a tree classifier, training data, validation/test/scoring data with a column titled "Customer", the two observations of interest, and the desired number of important features. The package produces a Waterfall Chart. 

**Examples**

**packages**

import pandas as pd

import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, RandomizedSearchCV

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


 
