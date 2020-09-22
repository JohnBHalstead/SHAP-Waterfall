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

**Scikit-Learn WI Breast Cancer Data Example**
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

**University of California, Irvine House Votes 84 data**

**packages**

import pandas as pd

import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, RandomizedSearchCV

import ship-waterfall

**models**

rf_clf = RandomForestClassifier()

**UCI Data **

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')

names = ['Republican', 'handicap infants', 'water project', 'budget resolution', 'physician fee freeze', 'el salvador aide', 'school religious groups', 'anti satellite', 'nicaraguan contras', 'mx missle', 'immigration', 'synfuels', 'education spending', 'superfund', 'crime', 'duty free exports', 'south africa']

df.columns = names

df = df.replace(to_replace =["republican", "y"], value = 1) 

df = df.replace(to_replace =["democrat", "n", "?"], value = 0) 

label = df.iloc[:,0]

features = df.iloc[:,1:17]

**data splits**
X_tng, X_val, y_tng, y_val = train_test_split(features, label, test_size=0.33, random_state=42)

print(X_tng.shape)

print(X_val.shape)

**fit classifiers and measure AUC**

clf = rf_clf.fit(X_tng, y_tng)

pred_rf = clf.predict_proba(X_val)

score_rf = roc_auc_score(y_val,pred_rf[:,1])

print(score_rf, 'Random Forest AUC')

*0.99238683127572 Random Forest AUC*

**IMPORTANT: add a 'Customer' column to the val/test/score data**

X_val = pd.DataFrame(X_val)

X_val['Customer'] = X_val.index

print(X_val.shape)

**Use Case 3**

ShapWaterFall(clf, X_tng, X_val, 78, 387, 5)

ShapWaterFall(clf, X_tng, X_val, 387, 78, 7)

**Use Case 4**

ShapWaterFall(clf, X_tng, X_val, 253, 157, 5)

ShapWaterFall(clf, X_tng, X_val, 157, 253, 7)

**Authors**

John Halstead, jhalstead@vmware.com

Ravi Prasad K, rkondapalli@vmware.com

Rajesh Vikraman, rvikraman@vmware.com

Kiran R, rki@vmware.com

**References**

1) Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data]. Irvine, CA: University of California, School of Information and Computer Science.

2) Kaloyan Iliev and Sayan Putatunda, “SHAP and LIME Model Interpretability”, VMware EDA AA & DS CoE PowerPoint Presentation, Palo Alto, CA, USA, November 21, 2019.

3) Dr. Dataman, “Explain Your Model with the SHAP Values”, Medium: Towards Data Science, available at https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d, September 13, 2019.

4) Sean Gillies, “The Shapely User Manual”, Shapely 1.8dev documentation, available at https://shapely.readthedocs.io/en/latest/manual.html, July 15, 2020.

5) Ashutosh Nayak, “Idea Behind LIME and SHAP: the intuition behind ML interpretation models”, Medium: Towards Data Science, available at https://towardsdatascience.com/idea-behind-lime-and-shap-b603d35d34eb, December 22, 2019.

6) Christoph Molnar, “Interpretable Machine Learning: a Guide for Making Black Box Models Explainable”, E-book available at https://christophm.github.io/interpretable-ml-book/, updated July 20, 2020, Chapters 5.7 (Local Surrogate (LIME)) and 5.10. (SHAP (SHapley Additive exPlanations)).

7) Scott Lundberg, “SHAP Explainers and Plots”, available at https://shap.readthedocs.io/en/latest/#, 2018.

8) Sean Owen, “Detecting Data Bias Using SHAP and Machine Learning: What Machine Learning and SHAP Can Tell Us about the Relationship between Developer Salaries and the Gender Pay Gap”, Databricks, available at https://databricks.com/blog/2019/06/17/detecting-bias-with-shap.html, June 17, 2019.

9) Chris Moffit, “Creating a Waterfall Chart in Python”, Practical Business Python, available at https://pbpython.com/waterfall-chart.html, November 18, 2014.

10) Abhishek Sharma, “Decrypting your Machine Learning model using LIME: why should you trust your model?”, Medium: Towards Data Science, available at: https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5, November 4, 2018.

11) Marco Tulio Ribeiro, “LIME Documentation, Release 0.1”, available at https://buildmedia.readthedocs.org/media/pdf/lime-ml/latest/lime-ml.pdf, August 10, 2017.

12) Lars Hulstaert, “Understanding model predictions with LIME”, Medium: Towards Data Science, available at https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b, July 11, 2018.

13) Ando Saabas, “treeinterpreter 0.2.2”, PyPl, available at https://pypi.org/project/treeinterpreter/, July 22, 2015.

14) Ando Saabas, “Random forest interpretation with scikit-learn”, Diving into Data: A blog on machine learning, data mining and visualization, available at http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/, August 12, 2015.

15) Manpreet Singh, Kiran R, and Stephen Harris, “Corona Impact: VMW Bookings and Propensity Models”, Vmware EDA AA & DS CoE PowerPoint Presentation, Palo Alto, CA, USA, 2019.

16) Scott M. Lundberg, Su-In Lee, “A Unified Approach to Interpreting Model Predictions”, 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA, 2016.

17) Lundberg, S., Lee, S., “A Unified Approach to Interpreting Model Predictions”, 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. 

18) Bowen, D., Ungar, L., “Generalized SHAP: Generating multiple types of explanations in machine learning”, Pre-print, June 15, 2020.

19) Veder, K., “An Overview of SHAP-based Feature Importance Measures and Their Applications To Classification”, Pre-print, May 8, 2020.

20) Lundberg, S., Erion, G., Lee, S., “Consistent Individualized Feature Attribution for Tree Ensembles”, Pre-print, March 7, 2019.
 
