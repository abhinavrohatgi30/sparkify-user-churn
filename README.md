# Sparkify : Predict User Churn for a Music Streaming Service

An Apache Spark ML Project to predict user churn for a music streaming service by developing a classifier using multiple features extracted from the user activity logs

## Overview
**Sparkify** is a digital music service. Many of the users stream their favorite songs in Sparkify service everyday, either using free tier that places advertisements in between the songs, or using the premium subscription model where they stream music as free, but pay a monthly flat rate. User can upgrade, downgrade or cancel their service at anytime.  

So, our job is mine the customers' data and implement appropriate model to predict customer churn as follow steps:

- Clean data: fill the nan values , correct the data types, drop the outliers.
- EDA: exploratory data to look features' distributions and correlation with key label (churn).
- Feature engineering: extract and found customer-features and customer-behavior-features; Implement standscaler on numerical features.
- Train and measure models:  I choose logistic regression, random forest classifier and gradient bossted tress classifier to train a baseline model and tuning a better model from best of them. It is worth mentioning that this data is unbalanced because of less churn customers, so we choose `f1 score`  as a metrics to measure models' performance.

## Installation

- Apache Spark 2.x
- Python 3.5+
- PySpark ML
- Jupyter
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Features used in the Models

Churn is defined as Cancellation Confirmation events in medium_sparkify_event_data.json data.

Following 11 features are defined to build models

1. Average listened songs per session
2. Listened total songs by users
3. Number of Add Friend transactions
4. Number of Add Playlist transactions
5. Number of Thumbs Down transactions
6. Number of Thumbs Up transactions
7. Register duration (days) - between last event date of user and registration date.
8. Length of listen time
9. Gender
10. Account level
11. Downgraded Event


## Results

The accuracy and F1 score of the three classification models used : Logistic Regression, Random Forest Classifier, Gradient Boosted Trees are,

Logistic Regression Metrics:
Accuracy of model is : 0.7443609022556391
F1 score of model is :0.6724855617304131
The training process of model took 13.907336235046387 seconds

Random Forest Metrics:
Accuracy of model is : 0.7969924812030075
F1 score of model is :0.7585255822483037
The training process of model took 12.764495134353638 seconds

Gradient Boosted Trees Metrics:
Accuracy of model is : 0.8270676691729323
F1 score of model is :0.799474962304081
The training process of model took 31.332529306411743 seconds

Why Gradient Boosted Model is best for our scenario?
First of all we used f1-score to select our final model because our problem is classification one and f1-score can help us to find the balance between accuracy and recall. Higher the f1-score the more perfect our model will be as false negative and false positive will be less.

Here Gradient Boosted Trees has best f1-score that's why I choose it for the next steps.


On futher hyperparameter tuning of the GBTClassifier we get the following metrics,

Gradient Boosted Trees Model - Test Metrics with best parameters:
Accuracy of model is :  0.8287671232876712
F1 score of model is : 0.8184619228046801

This is our final model and will be used to predict the user churn for this streaming service.


## References

Dataset provided by [Udacity](https://cn.udacity.com/).
