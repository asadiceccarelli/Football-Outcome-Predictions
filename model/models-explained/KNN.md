# K-Nearest Neighbours Classifier

## Theory

The KNN algorithm works to classify new data based on its proximity to the K nearest data points (neighbours). This calculates and average of their respective regression values (for regression), or a majority voting (for classification).

<p align='center'>
  <img 
    width='400'
    src='images/KNN.png'
  >
</p>

In the example above, the three nearest data points are included in the neighbourhood of the starred data as ```K=3```. Since the majority of these data points within this boundary are Class B, the starred data is included in Class B.

Limitations include:
- Complexity of algorithm dominated by having to calcualte the distance between each point and every other point int he neighbourhood
- Neighboring points may be close in feature space, but not in label space e.g. if they have similar values for features that do not influence the output label
- The whole dataset must be stored in the model when making predictions

## Hyperparameters to be tuned

1.	```n_neighbors```
    - The number of neighbours
2. ```p```
    - Power parameter for the Minkowski metric
    - ```p=1``` for Manhattan distance, ```p=2``` for Euclidean

## K-Fold Validation & Grid Search

Grid Search and K-Fold Validation are algorithmic ways to find hyperparameters. Grid search consists of a grid of possible hyperparameters with each combination used to tune the algorithm of choice and validate the results.

The main downside of Grid Search is that information on previous trials is not stored, and therefore every combination of hyperparameters are tried leading to long computation times. This makes it impossible to use with a large hyperparameter space.

K-Fold splits the dataset into multiple parts which are in turn used for training and validation.

<p align='center'>
  <img 
    width='400'
    src='images/k-fold.png'
  >
</p>

Algorithmically this goes:
1. Split data into ```K``` parts
2. Set ```i=0```
3. Take parts ```[i]``` as the validation set and the rest as the training set
4. Train the model with the training set
5. Calculate and save metrics on validation set
6. Increment ```i``` and repeat until last section is used as the validation set
7. Take the mean of the validation results

K-Fold is the de facto standard in machine learning as evaulation on single validation sets can be very noise leading to a wide variation in results. This gives us false impressions about the model performance as it usually does worse than expected. On the other hand, it is not standard in the deep learning community due to the amount of time K-Fold can take. E.g. for 5 splits, 5 separate models must be trained which is infeasible for larger models which may take weeks to train.

Grid Search and K-Fold can be used in conjunction with eachother.
