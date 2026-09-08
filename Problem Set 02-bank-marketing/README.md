Bank Term Deposit Prediction using Logistic Regression
Project Overview

This project uses Logistic Regression to predict whether a bank customer will subscribe to a term deposit (y = 1) or not (y = 0), based on their demographic details, account balance, and history with the bank's past marketing campaigns.

==========================================================================================================================================
Dataset
Name: Bank Marketing Data Set (bank-full.csv)
Rows: 45,211
Features: 16 predictive attributes — 7 numeric, 9 categorical
Target: y (binary — yes / no)
==========================================================================================================================================

Methodology
1. Loading & Inspecting the Data
Loaded bank-full.csv into the notebook and checked its shape, column types, and missing values.
Confirmed the file uses semicolons (;) as the delimiter rather than commas.
2. Preprocessing & Feature Engineering
Target encoding: Converted y from text to numbers (yes → 1, no → 0) so the model could use it.
Categorical encoding: One-hot encoded the 9 categorical columns (job, marital, education, housing, loan, contact, month, poutcome, default) using pd.get_dummies(..., drop_first=True), expanding the feature set to 42 columns.
Feature scaling: Standardized the numeric columns (age, balance, day, duration, campaign, pdays, previous) with StandardScaler, since their raw values were on very different scales and were causing the optimizer to fail to converge.
3. Training & Evaluation
Split: 80/20 train-test split (36,168 training rows, 9,043 test rows), using random_state=42 for reproducibility.
Model: LogisticRegression with max_iter=2000 to guarantee full convergence after scaling.
Metrics: Accuracy, confusion matrix, and a full classification report (precision, recall, F1-score).

==========================================================================================================================================

Results
Overall accuracy: ~89.88%
But accuracy alone is misleading here — the dataset is heavily imbalanced, with about 88% "no" and only 12% "yes." A model that always predicted "no" would already score close to 88%, so the real test is how well it catches actual subscribers.
Precision vs. recall gap: Precision for the majority class ("no") is high at 92%, but recall for the minority class ("yes") is only around 34% at the default 0.5 threshold — meaning the model misses roughly two-thirds of the customers who would actually subscribe.
What's driving predictions: Call duration and a successful outcome in the previous campaign (poutcome_success) stand out as the strongest predictors of a "yes."

==========================================================================================================================================
Takeaways
Scaling the numeric features fixed the lbfgs convergence warning and made training more stable.
High accuracy on an imbalanced dataset can hide a model that's actually performing poorly on the class that matters most — recall and precision on the minority class tell a more honest story.
A logical next step would be handling the imbalance directly (e.g. class_weight='balanced') or adjusting the classification threshold to improve recall on likely subscribers, even at some cost to overall accuracy.


==========================================================================================================================================
==========================================================================================================================================
