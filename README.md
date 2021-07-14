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

It the Scikit-learn Pipeline, I used LogisticRegression classification algorithm from sklearn library. 2 parameters have been adjusted through HyperDrive: inverse of regularization strength (parameter C) and Maximum number of iterations taken for the solver to converge (max_iter).

The larger the regularization strength (smaller values of C parameter) the higher the penalty for increasing the magnitude of parameters. This is to prevent overfitting the model to the train data and make it more general i.e. also applicable to the unseen test data.

In the HyperDrive run, max_iter parameter was selected out of 3 values (100, 200, or 400) and the C parameter was pooled from the interval between exp(-10) to exp(10) using logunifrom distribution. Since the parameter C represents the inverse of the regularization strength, logunifrom distribution has been selected to attain distribution of regularization strength as close to uniform as possible.

For early stopping policy, I selected a BanditPolicy class. This policy compares the current training run with the best performing run and terminates it if it’s performance metric drops below calculated threshold. The parameters I used in my pipeline:
```
policy = BanditPolicy(slack_factor=0.1, evaluation_interval=5, delay_evaluation=10)
```
would cause each run to be compared with the best performing run after each 5 algorithm runs (starting after first 10 runs) and if the run’s performance drops below 90% of current best run performance, then it would get terminated.

The best run was achieved with C= 4.217379487255968 and max_iter=400 with reported Accuracy of 0.9065.


## AutoML

The best performing model in the AuroML run was VotingEnsemble with reported 0.9021 accuracy. According to the best performing model, top 4 features (columns) with the largest importance were: nr.employed, cons.conf.idx, eurobor3m, contact_cellular
![image](https://user-images.githubusercontent.com/77756713/125700079-0832e013-d41e-4041-9fcd-0499aeee78c3.png)


## Pipeline comparison

The reported accuracy of both Scikit-learn Pipeline (0.9065) and Auto ML (0.9021) pipelines are very similar with no practical difference between their performance.

## Future work

The dataset is heavily imbalanced with nearly 90% of the values being "No" (29,258 vs 3,692 "Yes" answers). This could lead to false high model accuracy as even dummy model that always predicts "No" would be correct in 88% of the cases.

```
sns.countplot(y)
```
![image](https://user-images.githubusercontent.com/77756713/125700311-79a3027a-d274-48ce-b7e0-dff9fe3d9573.png)

Model run in the separate Notebook with the same Hyperparameters as the best performing Scikit-learn Pipeline model reported slightly lower model accuracy (0.9018) but the analys of the precision and recall values for 0 ("No") and 1 ("Yes") highlighted the problem.

the model is relatevely good at not labeling negative samples as positive (precision of "No" equal 0.91) and labeling negative samples as negative (recall of "No" equal 0.99) but not great at not labeling positive samples as negative (only 0.68) and very poor at finding all positive samples (recall of only 0.21). 


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
It would seem, that because of the unbalanced dataset, model is great at predicting "No"s but not great at predicting "Yes" answers. Next generation of this model would have to tackle this problem.

## Proof of cluster clean up

I have used the following code at the end of my run to ensure deletion of the computation cluster:
```
cpu_cluster.delete()
```

