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


| Model                  | Accuracy | F1-score | Training Time  |
|------------------------|----------|----------|----------------|
| Logistic Regression    | 0.744    | 0.672    | 13.9 s         |
| Random Forrest         | 0.796    | 0.758    | 12.76 s        |
| Gradient Boosted Trees | 0.827    | 0.799    | 31.33 s        |


Here Gradient Boosted Trees has best f1-score that's why I choose it for the next steps.


On futher hyperparameter tuning of the GBTClassifier we get the following metrics,
| Model                  | Accuracy | F1-score |
|------------------------|----------|----------|
| Gradient Boosted Trees | 0.828    | 0.818    |

This is our final model and will be used to predict the user churn for this streaming service.

## References

You can find a summarised analysis here [Medium](https://abhi-rohatgi.medium.com/sparkify-predicting-the-user-churn-using-apache-spark-ee4178f859c8)
Dataset provided by [Udacity](https://cn.udacity.com/).
