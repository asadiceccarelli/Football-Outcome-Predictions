# XGBoost Classifier

eXtreme Gradient Boosting, or XGBoost, refers to the speed enhancements such as parallel computing and cache awareness that makes it approximately 10 times fast than traditional Gradient Boosting. Not only this, but XGBoost includes a unique split-finding algorithm to optimise trees, along with built-in regularisation that reduces overfitting leading to it being a faster, more accurate version of Gradient Boosting. Since boosting usually performs better than bagging, and Gradient Boosting is arguably the strongest boosting ensemble, this makes XGBoost potentially the best supervised machine learning ensemble currently available.

<p align='center'>
  <img src='README-images/algorithm-comparison.png' width='450'>
</p>

> The evolution of XGBoost from Decision Trees.

## Theory

The algorithm works as follows:
1. Make an initial prediction
2. Build an XGBoost tree the predicts the residual of the initial prediction
3. Prune the XGBoost tree
4. Build another tree
5. Stop when a certain criteria has been met

## Hyperparameters to be tuned

1. `learning_rate`
        - Learning rate shrinks the contribution of each tree by `learning_rate`
        - A trade-off between `learning_rate` and `n_estimators`
2. `n_estimators`
        - The number of boosting stages to perform
        - Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance
3. `max_depth`
        - The maximum depth of the individual regression estimators
        - The maximum depth limits the number of nodes in the tree