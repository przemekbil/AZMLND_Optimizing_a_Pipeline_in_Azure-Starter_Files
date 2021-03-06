# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this project originated from the marketing campaigns of a Portuguese banking institution gathered over direct phone calls. The goal of this project is to predict whether client subscribed a term deposit (column "Y" value of "yes" or "no") using available independent variables.

The problem was solved using 2 different approaches: Logistic regression model which was optimised using HyperDrive Pipeline and with Azure AutoML.

Both methods yielded similar performance, with just over 90% accuracy. AutoML tried 50 models and the best performing one was the VotingEnsemble. 

## Dataset

The data used in the project has one dependent variable that we are trying to predict (column “Y”: client subscribed a term deposit) and the following groups of independent variables that will be used to build a model to predict value in column “Y” (as per https://archive.ics.uci.edu/ml/datasets/Bank+Marketing# ).
- Bank client data
- Last contact information of the current campaign
- Social and economic context information
- Information related to previous campaigns with the same client

Initial dataset analysis showed that it has 32,950 rows and doesn’t have any missing values. It’s worth mention, that the downloaded dataset size is smaller than the example dataset on UCI page, which has 41,188 rows. The filename in the provided azure blob location suggested, that the data has been already split into the train, test, and validate subsets and surely enough, the following locations provided missing rows of data:

https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_test.csv
https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_validate.csv

In both experiments runs (HyperDrive pipeline and AutoML) only bankmarketing_train.csv dataset was used. It was subsequently divided into train and test subsets using default 80-20 ratio.

Out of 20 independent variables, 10 turned out to be numerical and 10 categorical. Machine learning algorithms can only use numbers as the inputs, all the categorical variables were replaced by numbers in the clean_data function of the train.py script.
Additionally, column “Duration” has been removed from the dataset as it’s not known before the call is made. At the end of the call, when “Duration” is known so is the answer to question that we trying to predict. Therefore, this variable can’t be used in viable prediction model.

## Scikit-learn Pipeline

### Alghoritm

It the Scikit-learn Pipeline, I used LogisticRegression classification algorithm from sklearn library. 2 parameters have been adjusted through HyperDrive: inverse of regularization strength (parameter C) and Maximum number of iterations taken for the solver to converge (max_iter).

The larger the regularization strength (smaller values of C parameter) the higher the penalty for increasing the magnitude of parameters. This is to prevent overfitting the model to the train data and make it more general i.e. also applicable to the unseen test data.

### Parameter sampler

There are 2 parameters sweeping modes available in Azure ML: Entire grid and Random Sweep. Entire grid sweep is exhaustive, tends to give slightly better results but is time consuming. Random sweep in contrast offers good results without taking as much time. For this project, I'm going for the Random sweep using ```RandomParameterSampler``` Class using the following search spaces:

* max_iter parameter will be selected out of 3 predefined values (100, 200, or 400) 
* C parameter will be randomly pooled from the interval between exp(-10) to exp(10) using logunifrom distribution. Since the parameter C represents the inverse of the regularization strength, logunifrom distribution has been selected to attain distribution of regularization strength as close to uniform as possible.

### Termination policy

There are 3 early termination policies in Azure ML that can bu used to terminate runs with poor performing hyperparameters combinations and save computational resources:
* MedianStoppingPolicy - Defines an early termination policy based on running averages of the primary metric of all runs
* TruncationSelectionPolicy - Defines an early termination policy that cancels a given percentage of runs at each evaluation interval
* BanditPolicy - Defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation
* NoTerminationPolicy class can be used to specify that no early termination policy will be applied

For early stopping policy, I selected a BanditPolicy class. This policy compares the current training run with the best performing run and terminates it if it’s performance metric drops below calculated threshold. The main benefit of using this policy comparing to the other 2 policies is that the current runs are terminated after comparing with the best performing run. If the current run performance drops greatly below the best run's performance, it will be terminated. 

The parameters I used in my pipeline:
```
policy = BanditPolicy(slack_factor=0.1, evaluation_interval=5, delay_evaluation=10)
```
would cause each run to be compared with the best performing run after each 5 algorithm runs (starting after first 10 runs) and if the run’s performance drops below 90% of current best run performance, then it would get terminated.

### Performance

The best run was achieved with C= 4.217379487255968 and max_iter=400 with reported Accuracy of 0.9065. The confusion matrix for this run is as follows:

![image](https://user-images.githubusercontent.com/77756713/125952629-a3c44167-472c-4d61-a6f1-6bc300b02508.png)

From the confusion matrix above, we can see that the model is biased towards "No" answers: it is very good at predicting "0" not only when the real label is "0" (98.73%) but also when the real label should be "1" (79.25% "Yes" answers has been predicted as "No"). It's very bad however at predicting "1s": only 20.75% "Yes" answers were predicted correctly.

I have compared this model's performance with a Dummy classifier from ```sklearn.dummy``` library:
```
dummy_clf=DummyClassifier(strategy="most_frequent")

dummy_clf.fit(x_train, y_train)

print("Dummy clasifier accuracy: ",dummy_clf.score(x_test, y_test))
Dummy clasifier accuracy:  0.8841951930080116
```

Although the accuracy of my best model (0.9065) seems good on the face value, it is only marginally better than the Dummy model (0.8841) which is not great news considering how much effort was put into developing it. The most likely reason for it's poor performance and unexpectedly high accuracy of the Dummy classifier is the fact that the data set is heavy imbalanced towards "No" (more about this in "Future Work" paragraph below).


## AutoML

Great function of AutoML is the automatic check for the issues with the data set used for training. This is to alert the user about the potential model inaccuracies and to allow to take the corrective actions. In my example, I was alerted about the problem with Class balancing: the size of the "Yes" class is only 2738, which is about 11% of the total data set.

The other 2 check for "Missing feature values imputation" and "High cardinality feature detection" passed without any alerts.

The highest scoring model in the AutoML run was VotingEnsemble with reported 0.9021 accuracy. The VotingEnsemble was constructed out of previous best performing unique runs, with individual weights assigned to each of them to optimize the overall performance. The inner estimators, their weights, Iteration number and original metrics are listed in the table below:


ITERATION | Algorithm | Weight | Metric |
--- | --- | --- | --- |
1 | maxabsscaler, xgboostclassifier | 0.4 | 0.9017 | 
42 | sparsenormalizer,	lightgbmclassifier |	0.066666667 |	0.9011 |
24|	sparsenormalizer,	lightgbmclassifier|	0.066666667|	0.9009|
28|	maxabsscaler,	lightgbmclassifier|	0.066666667|	0.9009|
40|	standardscalerwrapper,	xgboostclassifier|	0.066666667|	0.9009|
54|	maxabsscaler,	lightgbmclassifier|	0.133333333|	0.9007|
48|	sparsenormalizer,	xgboostclassifier|	0.133333333|	0.8999|
0|	maxabsscaler,	lightgbmclassifier|	0.066666667|	0.8999|



According to the best performing model, top 4 features (columns) with the largest importance were: nr.employed, cons.conf.idx, eurobor3m, contact_cellular

![image](https://user-images.githubusercontent.com/77756713/125700079-0832e013-d41e-4041-9fcd-0499aeee78c3.png)


The confusion table for this model is presented below:

![image](https://user-images.githubusercontent.com/77756713/125952917-8c13085b-c495-41bd-97b2-918bf462659c.png)

Similarly to the model developed through the Scikit-learn Pipeline, AutoML model is great at predicting "No's" (98.58% of the "No" answers were predicted correctly) but is doesn't do great job at correctly predicting "Yes" answers: 76.44% percent of the original "Yes" answers were predicted as "No".



## Pipeline comparison

### Performance

The reported accuracy of models developed through Scikit-learn Pipeline (0.9065) and Auto ML (0.9021) pipelines are very similar with no practical difference between their performance and are just over the accuracy of the Dummy classifier. Both pipelines handled poorly imbalanced data set and produced models that are heavily skewed towards predicting "No" answers.

### Model acquisition

Through the Scikit-learn Pipeline we are using only one algorithm which we optimize using different hyperparameters: regularization strength (parameter C) and Maximum number of iterations taken for the solver to converge (max_iter).

AutoML on the other hand, used 13 different algorithms (including LogisticRegression, although non of it's run's made it to top 15) in 57 separate runs executed with different hyperparameters and the final model was constructed as a combination of 8 best performing runs. This is typically more robust approach, which tends to help to improve model generalization over a single estimator.


## Future work

The dataset is heavily imbalanced with nearly 90% of the values being "No" (29,258 vs 3,692 "Yes" answers). This could lead to false high model accuracy as even dummy model that always predicts "No" would be correct in 88% of the cases.

```
sns.countplot(y)
```
![image](https://user-images.githubusercontent.com/77756713/125700311-79a3027a-d274-48ce-b7e0-dff9fe3d9573.png)

Model run in the separate Notebook with the same Hyperparameters as the best performing Scikit-learn Pipeline model reported slightly lower model accuracy (0.9018) but the analysis of the precision and recall values for 0 ("No") and 1 ("Yes") confirmed the problem.

The model is relatively good at not labelling negative samples as positive (precision of "No" equal 0.91) and labelling negative samples as negative (recall of "No" equal 0.99) but not great at not labelling positive samples as negative (only 0.68) and very poor at finding all positive samples (recall of only 0.21). 


```
model = LogisticRegression(C=4.217379487255968, max_iter=400).fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("The model accuracy is :", accuracy)

y_predict = model.predict(x_train)
print(classification_report(y_train, y_predict))

The model accuracy is : 0.9017965525613013
              precision    recall  f1-score   support

           0       0.91      0.99      0.95     21939
           1       0.68      0.21      0.32      2773

    accuracy                           0.90     24712
   macro avg       0.79      0.60      0.63     24712
weighted avg       0.88      0.90      0.88     24712
```
It would seem, that because of the unbalanced dataset, model is great at predicting "No"s but not great at predicting "Yes" answers. 

To achieve better performing model, data used for training  should be balanced by either up-sampling the "Yes" Class or down-sampling the "No" class. Alternatively, wight column Could be added to cause rows in the data to be weighted up or down (to make the prevailing class seem less important).

Another thing to try would be using other ```primary_metric``` attribute than ```accuracy``` in the ```AutoMLConfig``` class constructor. According to Microsoft AutoML documentation, ```AUC_weighted``` could be better choice in this case.

Depending on the real-life application, we might want to optimize our model for different metrics: if our priority is model's precision (avoiding labeling false positives and false negatives) we could use ```average_precision_score_weighted``` or ```precision_score_weighted``` as our ```primary_metric``` attribute. 


## Proof of cluster clean up

I have used the following code at the end of my run to ensure deletion of the computation cluster:
```
cpu_cluster.delete()
```

