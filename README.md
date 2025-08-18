# Compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines)

### Cource : Module 17 Berkeley
Link to my notebook: https://github.com/Minasardari/Berkeley_Module17_MarketingCampaigns/blob/main/MinaSardari_MarketingCampaigns_Result.ipynb
### Business Goal : 
The business goal is to predict that if the client would subscribed a term deposit.
Type: binary classification, class-imbalanced (positives are usually scarce).
# **Busines Objective**:
The objective of this project is to improve the effectiveness of the bank’s direct marketing campaigns by predicting whether a client will subscribe to a term deposit, the article marntioned they data contain 17 campaigns information.

By analyzing client demographics, financial attributes, past campaign interactions, and economic indicators, the bank wants to:

*   Identify the most promising clients who are likely to subscribe,
*   Optimize marketing resources (reduce wasted calls and time),
*   Increase conversion rates for term deposits, and
*   Gain insights into which client and campaign factors most influence success.

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
Handle duplicates, impossible ages outliers, negative counts, etc.
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



### 1. Understanding the Dataset

Step 1: Load and Inspect the Data
df.head() – View the first few rows to understand structure.
df.info() – Check column names, data types, and non-null counts.
df.describe() – Review statistical summaries of numeric columns.
df.columns – Look for naming inconsistencies

Step 2: Identify Missing Values
#### Calculate missing values and their percentage
 <img width="224" height="344" alt="image" src="https://github.com/user-attachments/assets/316d157a-e657-4281-bc57-1cc27c5b7896" />


Step 3: explore data uniqe values for categorial features 
  Calculate proportions
  Replace NaN index labels with a string for display ''unknown
  Plot to see value counts
<img width="454" height="277" alt="image" src="https://github.com/user-attachments/assets/0caba558-3686-4d99-acc6-43da9a2641f1" />
<img width="1187" height="1189" alt="image" src="https://github.com/user-attachments/assets/0d5da2b1-38c2-46ea-b69e-619e0e5328cf" />


### 2. Initial Data Cleaning
step 1: Drop Duplicate
Step 2: Drop Columns 
step 3: Categorical Features Imputation: Data provided is pretty clean and no NAN or it has been replaced by Unknown
step 4: Detect Data Quality Issues : Data was clean no inconsistent values in categorical columns
Step 5: Numerical Features Imputation: didn't find any column to impute for now later based on EDA we may do some more

### 3. EDA 
 #### a. Univariate Analysis
<img width="1187" height="1189" alt="image" src="https://github.com/user-attachments/assets/e161d056-a655-47a6-97ef-a605ddc7446a" />
<img width="1489" height="922" alt="image" src="https://github.com/user-attachments/assets/d8281218-3a71-41bd-ab06-d3ac867b651a" />
<img width="1489" height="922" alt="image" src="https://github.com/user-attachments/assets/9201e0fc-35b7-47a0-a091-7718762af173" />

 #### b. Bivariate and Multivariate Analysis

<img width="1990" height="2390" alt="image" src="https://github.com/user-attachments/assets/5b559f98-757b-41d3-8b59-791321a57984" />
<img width="1489" height="923" alt="image" src="https://github.com/user-attachments/assets/5d68bd04-88e4-4a62-b26e-a8f9e8e64383" />
<img width="589" height="569" alt="image" src="https://github.com/user-attachments/assets/0c9ad0bb-cba2-440d-a547-df478b737348" />
<img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/08529f2a-bd7a-41b0-a384-d85b2b7d3e80" />
<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/5bf0fa44-ef20-47d7-998c-3dd8f2bd9d61" />

#### c. Findings
<img width="686" height="395" alt="image" src="https://github.com/user-attachments/assets/70ea11bb-17f8-4b4d-a09d-2954d7eb0367" />

**Numerical:**
1. age : Right-skewed, most clients between 25–60, with outliers up to ~100. outlier >70 but may keep <90, maybe keep older clients’ signal as seems retired customer who say Yes are low amost 25% so amybe better keep 70 to 90.
 Distribution for yes and no is quite similar (both center ~38–40)
2. campaign:  
  *   Highly skewed. extreme outliers (values > 10).
  *   Median campaign is slightly higher for “no” (more calls → more rejection).
  *   Those who said “yes” usually had fewer contacts.

3. previous: 
  *   Mostly 0, very skewed. may onvert to a binary indicator (prev_contacted = previous > 0).
  *   Clients who subscribed (“yes”) had higher previous values (more often contacted before).

4. emp_var_rate: 
  *   Discrete values with clear peaks -3, -2, -1, 0, 1.
  *   lear difference:
   no -> higher employment variation rate (~1).
   yes -> lower, even negative values.

5. cons_price_idx: Clustered around specific values, looks more categorical.
    *    yes clients are slightly associated with lower price index (93.0–93.5) compared to no.
    *   Difference is modest but present.
6. cons_conf_idx: 
    *   Distinct clusters/peaks.
    *  yes clients linked to lower consumer confidence (worse sentiment, ~–40 vs –45).
    *  Good separation.
7. euribor3m:WiStrong separation:

    *  no clients -> higher rates (~5).

    *  yes clients -> lower rates (~1).
8. nr_employed:
    *  highly correlated with emp_var_rate and euribor3m
    *  no → higher employment numbers (~5200).
    *  yes → lower (~5000).


**categorical:**

1. job

  *   Diverse categories, with admin, blue-collar, technician, services dominating.
  *   Some categories have very few samples (e.g., unknown).
  *   Decision: grouping rare jobs into "Other".

2. marital

  *   Mostly married, then single, fewer divorced.
  *   Decision: Drop unknown (very few).

3. education

  *   Spread across several categories, with university degree leading.
  *   Some categories very small (illiterate, unknown).
  *   Decision: Group rare levels.

4. default

  *   Majority no, with many unknown. Very few yes.
  *   Decision: Potentially drop, since too imbalanced and dominated by unknowns.

4. housing  
  *   Balanced between yes and no, few unknown.
  *   Decision: keep maybe combine with loan to make onw has_loan
5. loan
  *   Mostly no, some yes, some unknown.
  *   Decision: keep maybe combine with housing to make onw has_loan

6. month
  *   imbalance
  *   Decision: Keep, but instead of raw categorical, encode as seasonality
7. day_of_week
  *   very balanced and also in yes percentage almost same for all days
  *   Decision: Drop low impact
8. poutcome
   *   Mostly nonexistent (no previous campaign), few success/failure.
  *   Decision: based on barchart yes percetage is high when previous campaingn is high
9. prev_contacted
   *   Mostly no, some yes.
   *   Decision: Keep — useful binary predictor.



**the correlation heatmap:**

*    Observations


1. High correlations ( 0.9+)
    emp_var_rate, euribor3m, and nr_employed are very strongly correlated (red blocks). This means they carry the same economic signal (labor market + interest rates). we can use **PCA**
2. Moderate correlations

      emp_var_rate also moderately correlated with cons_price_idx and cons_conf_idx.
      That makes sense: all are macroeconomic indicators.

3. Low/near-zero correlations

      age, campaign, and previous are not correlated with the economic features they bring independent customer info.

**PairPlot**
Age Distribution is skewed (most clients between 30–60).

### 5. Feature Engineering Ideas
**1. Age Binning**

  *    Capped ages at <90.
  *    Grouped ages into categories: <30, 30–50, 50–70, 70+.
  *    Dropped the original age column after creating age_group.

**2. Campaign Outliers**

  *    Limited (capped) campaign contacts to 10 (values above 10 set to 10).
**3. Previous Contacts**
  *    Converted previous into a binary indicator: 1 if previously contacted, 0 otherwise.

**4. Job Simplification**
  *   Grouped rare job categories (less than 1720 occurrences) into "Other".
  *   Dropped the original job column.

**5. Marital Status**
  *   Dropped rows with "unknown" marital status (very few cases).

**6. Education Simplification**
  *   Mapped education into broader groups:
      *    university.degree / professional.course → Higher**
      *    basic.4y / basic.6y / basic.9y → Basic
      *    high.school → High School
      *    illiterate / unknown → Other
  *  Dropped the original education column.

**7. Default Column**
  *   Dropped default column (mostly "no" or "unknown", very few "yes").

**8. Loan Information**
  *   Combined housing and loan into a single column has_loan (1 if either is "yes").
  *   Dropped housing and loan.

**9. Day of Week**
  *   Dropped day_of_week since it was fairly balanced and not predictive.

**10. Previous Contacted Flag**
  *   Converted prev_contacted into binary: 1 = yes, 0 = no.

**11 Month**
  *   Created month_quarter feature (Q1–Q4).
  *   Dropped the raw month and month_num columns.

  **Final Features**

  <img width="1579" height="124" alt="image" src="https://github.com/user-attachments/assets/3e295a73-235e-4577-bf87-cf961dbc2f69" />

### 6. Test the null hypothesis:
 <img width="708" height="351" alt="image" src="https://github.com/user-attachments/assets/f9992d41-5345-4c7e-8f8e-6c87945352fa" />


### 7. Model Building

     **Pre-processing** 
     
     * Train - Test Split
     * One-Hot encoding 
     * StandardScaler - Numerical 

   *   **A Baseline Model**
       Before we build our first model, we want to establish a baseline. What is the baseline performance that our classifier should aim to beat?
       Baseline Model (Dummy Classifier)

          **Model:**
          Used DummyClassifier(strategy="most_frequent") → always predicts the majority class from the training set.
          
          **Purpose:**
          Provides a baseline that real models should beat. If your ML models don’t outperform this, they’re not useful.
          
          **Metrics Evaluated:**
          
          Accuracy (train/test) – how often the classifier is correct.
          ROC-AUC – area under the ROC curve (default is 0.5 if no probability output).
          F1 (positive class) – harmonic mean of precision and recall for the yes class.
          PR-AUC (Average Precision) – area under the precision–recall curve, useful for imbalanced data.
          
          **Implementation Notes:**
          Probabilities (predict_proba) are used if available; otherwise, the code falls back to class predictions.
          ```python
          Baseline:
          accuracy train: 0.8875235719934302
          accuracy test: 0.8874695863746959
          roc_auc: 0.5
          f1_positive: 0.0
          pr_auc: 0.11253041362530414
          ```
   *   **A Simple Model**
          Use Logistic Regression to build a basic model on your data.
      **Model:**
         Trained a LogisticRegression classifier on the processed dataset.
     **Purpose:**
         Serves as the first real predictive model beyond the baseline. It tests whether linear decision boundaries can separate the target classes.
     **Metrics Evaluated:**
         Accuracy (train/test) – proportion of correct predictions.
         ROCAUC – area under the ROC curve, capturing the model’s ability to rank positive vs negative cases.
         F1 (positive class) – harmonic mean of precision and recall for the yes class.
         PR-AUC (Average Precision) – area under the precision–recall curve, helpful for imbalanced classes.
      ```python
          Logistic Regression:
          accuracy train: 0.8092341383295821
          accuracy test: 0.8104622871046229
          roc_auc: 0.7896102106218623
          f1_positive: 0.4375451263537906
          pr_auc: 0.4491645267848371
          time_inference_sec: 49.91781949996948
       ```
   
       Get Top features (by |coef|):
       ```python
                      feature  importance
                 emp_var_rate    1.584100
             month_quarter_Q4    1.523706
                    euribor3m    1.390182
             month_quarter_Q2    1.363378
                  nr_employed    0.834115
             month_quarter_Q3    0.772944
             poutcome_success    0.739794
            contact_telephone    0.550660
                age_group_70+    0.441395
               cons_price_idx    0.354817
                age_group_<30    0.230109
               prev_contacted    0.212883
       job_grouped_bluecollar    0.197594
                     previous    0.177836
         job_grouped_services    0.166620
              age_group_50-70    0.166039
      education_grouped_Other    0.138520
       job_grouped_management    0.125899
     education_grouped_Higher    0.122312
       job_grouped_technician    0.093119
               marital_single    0.089785
            job_grouped_admin    0.081088
                     campaign    0.063003
                cons_conf_idx    0.061840
education_grouped_High School    0.037251
              marital_married    0.027753
                     has_loan    0.018548
         poutcome_nonexistent    0.011729
     ```


     ### **Dummy Classifier (predicts majority class or random)**
       
       *   Accuracy train/test ≈ 88.7% (same as majority “no” rate).
       *   ROC-AUC = 0.5 Random guess    
       *   F1 = 0.0 ->never predicts positive
       *   PR-AUC ≈ 0.11 -> equal to positive class prevalence.
        
       **⚠️This shows what happens if you don't learn anything always predict no.**

      ### **Logistic Regression (baseline model)**

       *   ROC-AUC = 0.79 -> strong improvement, the model is actually ranking positives higher.
       *   PR-AUC = 0.45 -> way above the 0.11 baseline someans model is capturing meaningful signal.
       *   F1(0.5) = 0.439 -> good balance between precision and recall.
       *   Training time = 0.04 sec -> fast and efficient.

       ✅**Logistic Regression is clearly learning patterns that separate “yes” vs. “no”, even though the dataset is imbalanced.**

#### 8. Model Evaluation 
#### Compare Evaluate Models for classifiers
Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, fit and score each. Also, be sure to compare the fit time of each of the models. Present your findings

<img width="952" height="525" alt="image" src="https://github.com/user-attachments/assets/9e062493-4f05-46c1-9604-d635c931660e" />
**Key Observations:**

| Model                  | Train Time (s) | Train Accuracy | Test Accuracy |
| ---------------------- | -------------- | -------------- | ------------- |
| Logistic Regression    | 46.40          | 0.8092         | 0.8105        |
| K-Nearest Neighbors    | 0.12           | 0.9124         | 0.8942        |
| Decision Tree          | 0.22           | 0.9714         | 0.8634        |
| Support Vector Machine | 11.44          | 0.8875         | 0.8875        |


Logistic Regression: Stable and balanced performance (train ≈ test).
KNN: Very fast, strong performance but may overfit slightly (train > test).
Decision Tree: Extremely high training accuracy but weaker test accuracy → clear overfitting.
SVM: Solid balance, with good generalization and moderate training time.

#### Improving the Model
Now that we have some basic models on the board, we want to try to improve these. Below, we list a few things to explore in this pursuit.

More feature engineering and exploration. 
Hyperparameter tuning and grid search. All of our models have additional hyperparameters to tune and explore. For example the number of neighbors in KNN or the maximum depth of a Decision Tree.
Adjust your performance metric
1. Appled pCA but model had issue so revert
2. After fitting a Logistic Regression model, we analyzed feature importance using the absolute values of the coefficients. Features with very small weights (low predictive impact) were identified and dropped:
Dropped Features:  campaign (coef ≈ 0.061) cons_conf_idx (coef ≈ 0.043)
3. Find best Hyper Parameter using GridSearchCV


| Model                   | Best Params                                                                                           | Train Time (s) | CV ROC-AUC (best) | Train Acc | Test Acc | Test ROC-AUC | Test PR-AUC | Test F1\@0.5 |
| ----------------------- | ----------------------------------------------------------------------------------------------------- | -------------- | ----------------- | --------- | -------- | ------------ | ----------- | ------------ |
| **Logistic Regression** | {`lr__C`: 10, `lr__penalty`: l2, `lr__solver`: lbfgs}                                                 | 69.435         | 0.7868            | 0.8097    | 0.8106   | 0.7896       | 0.4491      | 0.4377       |
| **Decision Tree**       | {`dt__criterion`: entropy, `dt__max_depth`: 5, `dt__min_samples_leaf`: 5, `dt__min_samples_split`: 2} | 9.373          | 0.7823            | 0.9018    | 0.9024   | 0.7893       | 0.4354      | 0.3725       |
| **SVM**                 | {`clf__C`: 10, `clf__gamma`: scale, `clf__kernel`: linear}                                            | 250.830        | 0.7705            | 0.8940    | 0.8974   | 0.7871       | 0.4396      | 0.3704       |
| **KNN**                 | {`knn__metric`: manhattan, `knn__n_neighbors`: 11, `knn__weights`: uniform}                           | 37.199         | 0.7537            | 0.9069    | 0.8987   | 0.7612       | 0.3828      | 0.3656       |



### Observations

- **Logistic Regression**
  - Balanced and stable performance (train ≈ test).  
  - Best **PR-AUC** (0.4491) and **F1** (0.4377), making it the strongest model for imbalanced positive detection.  

- **Decision Tree**
  - Very fast training with high accuracy (0.9024 test), but **PR-AUC and F1 are weaker** than Logistic Regression → suggests bias toward the majority class.  
  - Slight overfitting risk (train > test).  

- **SVM**
  - Achieved strong accuracy (0.8974) and competitive ROC-AUC (0.7871).  
  - Training was **computationally expensive** (250s).  
  - PR-AUC and F1 lower than Logistic Regression → weaker on positives.  

- **KNN**
  - Highest training accuracy (0.9069), good test accuracy (0.8987).  
  - But **lowest ROC-AUC, PR-AUC, and F1**, confirming it struggles with imbalanced class separation.  
  - Moderate training cost compared to SVM.  

**Conclusion:**  
Logistic Regression, despite being the simplest model, provides the best **balance of generalization, interpretability, and positive-class detection**.

But lowest ROC-AUC, PR-AUC, and F1, confirming it struggles with imbalanced class separation.

Moderate training cost compared to SVM.

Conclusion: Logistic Regression, despite being the simplest model, provides the best balance of generalization, interpretability, and positive-class detection.
  
