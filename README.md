# SHAP-Waterfall

**Install**

Using pip (recommended)
    
    pip install shap-waterfall
    
**Introduction**

Many times when VMware Data Science Teams present their ML models' propensity to buy scores (estimated probabilities) to stakeholders,  stakeholders ask why a customer’s propensity to buy is higher than the other customer. Their question was our motivation.  Plus recent EU and US mandates that direct machine learning model explainability became another motivation.

This graph solution provides a local classification model interruptibility between two observations, which we call customers. It uses each customer's estimated probability and fills the gap between the two probabilities with SHAP values that are ordered from higher to lower importance.

Currently, this package only works for tree and tree ensemble classification models. Our decision to limit the use to tree methods was based on two considerations. We desired to take advantage of the tree explainer's speed. As a business practice, we tend to deploy Random Forest, XGBoost, LightGBM, and  other tree ensembles more often than other classifications methods.

However, we plan on including the kernel explainer in future versions.

The package requires a tree classifier, training data, validation/test/scoring data with a column titled "Customer", the two observations of interest, and the desired number of important features. The package produces a Waterfall Chart. 

**Examples**



 
