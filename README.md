# Berkeley_Module17_MarketingCampaigns

# Compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines)
### Cource : Module 17 Berkeley
Link to my notebook: https://github.com/Minasardari/Berkeley_Module17_MarketingCampaigns/blob/main/MinaSardari_MarketingCampaigns_Result.ipynb
### business goal : 
The business goal is to predict that if the client would subscribed a term deposit.
Type: binary classification, class-imbalanced (positives are usually scarce).

Business Understanding → Data Science Perspective:
From a data science perspective, In this practical application, my goal is to compare the performance of the classifiers we have introdued (K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines). I will utilize a dataset related to marketing bank products over the telephone.

Using the CRISP-DM framework, this project will begin with exploratory data analysis (EDA) to uncover patterns, trends, and relationships within the dataset. This will be followed by data cleaning and feature engineering to enhance model performance. Ultimately, model will be developed and evaluated to with performance and qulaity measurement such as Recall, Accuracy to guide identify the new customer would or would not subscribed a term deposit.

### Project Roadmap (CRISP-DM Steps):

#### 1. Understanding the Dataset
Grasp the structure, variables, and potential quality issues within the dataset.

#### 2. Initial Data Cleaning
Address missing values, duplicates, and incorrect data types; prepare the data for analysis. 
Ensure types: categorical vs numeric; lower-case strings; strip whitespace.
Target map: y ∈ {yes,no} → {1,0}.
Specials : pdays=999 ⇒ “not previously contacted”. Create prev_contacted = (pdays != 999) and replace 999 with NaN (or keep two features).
Handle duplicates, impossible ages (e.g., < 16 or > 100), negative counts, etc.
#### 3. EDA – Univariate Analysis
Explore the distribution of individual variables to identify trends, outliers, and data quality issues.

#### 4. EDA – Bivariate and Multivariate Analysis
Investigate relationships between predictors and the target variable, as well as between predictors.

#### 5. Feature Engineering Ideas
Create or transform variables to better capture patterns and improve model performance.

#### 6. Behavioral Insights & Hypotheses
Formulate hypotheses based on data patterns — e.g.,  ""

#### 7. Model Building
Train multiple regression models (e.g., linear regression, decision trees, random forest) to predict car prices.

#### 8. Model Evaluation
Split: Stratified train/test (80/20).
Use StratifiedKFold for CV; report ROC-AUC, F1, Precision/Recall, Confusion Matrix.
#### 9. Data Visualization for Storytelling
Develop intuitive charts and dashboards to communicate findings and support data-driven decision-making for the bank.

Mina 5/23


