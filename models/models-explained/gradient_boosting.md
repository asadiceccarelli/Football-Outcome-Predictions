# Gradient Boosting

Instead of fitting a predictor on the data at each iteration (like AdaBoost), Gradient Boosting fits a new predictor to the residual errors made by the previous predictor. It is an iterative functional gradient algorithm, i.e an algorithm which minimizes a loss function by iteratively choosing a function that points towards the negative gradient.

Advantages:
- Accepts various types of inputs to make it more flexible
- Can be used for both regression and classifications
- Gives features important for the output

Disadvantages:
- Takes longer to train as cannot be parallelised
- More likley to overfit as is obsessed with the wrong output since it learns from past mistakes
- Tuning can be difficult due to the large number of parameters

## Theory

The algorithm works as follows:

1. First, the algorithm calculates the _log of the odds_ of the target feature. This is then converted to a probability using a logistic function in order to make initial predictions.

2. For each instance in the training set, the _residuals_ are calculated i.e. the observed value minus the predicted value.

3. A new decision tree that tries to predict the residuals that were previously calculated is built. For each instance in the training set, the formula for making predictions is:
```
base_log_odds + (learning_rate * predicted residual value)
```

4. This log(odds) prediction is converted into a probability.

5. New residuals of the tree are calculated and a new tree to fit these is created.

6. This process is repeated until a certain predefined threshold is reached.

## Hyperparameters to be tuned:

1. `loss`
        - The loss function to be optimized
        - `log_loss` refers to binomial and multinomial deviance, the same as used in logistic regression, and is often used for classification
        - `exponential` recovers the AdaBoost algorithm
2. `learning_rate`
        - Learning rate shrinks the contribution of each tree by `learning_rate`
        - A trade-off between `learning_rate` and `n_estimators`
3. `n_estimators`
        - The number of boosting stages to perform
        - Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance
4. `max_depth`
        - The maximum depth of the individual regression estimators
        - The maximum depth limits the number of nodes in the tree