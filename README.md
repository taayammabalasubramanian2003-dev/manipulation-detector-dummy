Earnings Manipulation Detection – ML Project
Project Overview

This project develops a Machine Learning model to identify firms that are likely to manipulate earnings using Beneish financial ratios. The objective is to support auditors, banks, regulators, and investors in detecting potential financial misreporting.

Dataset

The dataset contains the following variables:

DSRI – Days Sales Receivable Index

GMI – Gross Margin Index

AQI – Asset Quality Index

SGI – Sales Growth Index

DEPI – Depreciation Index

SGAI – SG&A Expense Index

ACCR – Total Accruals

LEVI – Leverage Index

Manipulator – Target variable (1 = Manipulator, 0 = Non-Manipulator)

Machine Learning Approach

Two classification models were developed:

Logistic Regression

CART (Decision Tree)

K-Fold Cross Validation and GridSearchCV were used to tune hyperparameters and select the best performing model based on Recall and ROC-AUC.



