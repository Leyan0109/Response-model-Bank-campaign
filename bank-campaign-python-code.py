#!/usr/bin/env python
# coding: utf-8

pip install scipy==1.11.4

# Importing essential libraries for data analysis, modeling, and visualization
import numpy as np  # NumPy for numerical operations and handling arrays
import pandas as pd  # pandas for data manipulation and analysis
import scipy.stats as stats  # SciPy for statistical functions
import matplotlib.pyplot as plt  # Matplotlib for data visualization
import seaborn as sns  # Seaborn for statistical plotting and making plots easier to generate
import statsmodels.formula.api as smf  # statsmodels for statistical modeling
from sklearn.model_selection import train_test_split  # Scikit-learn for splitting data into training and testing sets
from sklearn import metrics  # Scikit-learn for model evaluation metrics
get_ipython().system('pip install scikit-plot  # Installing the scikit-plot package for enhanced plotting of classification results')
import scikitplot as skplt  # scikit-plot for creating advanced plots like ROC and confusion matrices for model evaluation

# Reading the 'all_campaign.csv' file into a pandas DataFrame, while treating specified values ("NA", "N/A", "None", "null", " ") as missing values (NaN).
df1 = pd.read_csv('all_campaign.csv', na_values=["NA", "N/A", "None", "null", " "])

# Displaying the first 5 rows of the DataFrame to get an overview of the data
df1.head()

# Check for missing values in the entire DataFrame
print(df1.isnull().sum())

# Reading the 'all_personal.csv' file into a pandas DataFrame, while treating specified values ("NA", "N/A", "None", "null", " ") as missing values (NaN).
df2 = pd.read_csv("all_personal.csv", na_values=["NA", "N/A", "None", "null", " "])

# Displaying the first 5 rows of the DataFrame to get an overview of the data
df2.head()

# Check for missing values in the entire DataFrame
print(df2.isnull().sum())

# Filling missing values in the 'region', 'job', and 'education' columns with the string "Missing"
df2[['region','job', 'education']] = df2[['region','job', 'education']].fillna("Missing")

# Checking for any remaining missing values in the DataFrame by summing the null values across each column
print(df2.isnull().sum())

# Displaying the first 5 rows of the updated DataFrame to inspect the changes
df2.head()

# Merging df1 and df2 on the 'custID' column using an inner join to combine information from both DataFrames
df3 = pd.merge(df1, df2, how='inner', on='custID')

# Displaying the first 5 rows of the merged DataFrame to inspect the combined data
df3.head()

# Check for missing values in the entire DataFrame
print(df3.isnull().sum())

# Converting categorical 'yes' and 'no' values to numeric values (1 and 0) for the response and other related columns
df3['response'] = df3['response'].replace({'yes': 1, 'no': 0})  # Converting 'response' to binary values (1 for 'yes', 0 for 'no')
df3['default'] = df3['default'].replace({'yes': 1, 'no': 0})  # Converting 'default' to binary values (1 for 'yes', 0 for 'no')
df3['housing'] = df3['housing'].replace({'yes': 1, 'no': 0})  # Converting 'housing' to binary values (1 for 'yes', 0 for 'no')
df3['loan'] = df3['loan'].replace({'yes': 1, 'no': 0})  # Converting 'loan' to binary values (1 for 'yes', 0 for 'no')

# Displaying the first 5 rows to verify the changes
df3.head()
df3.info()

# Converting other columns to 'category' data type to optimize memory usage and processing time for categorical features.
df3['contact'] = df3['contact'].astype('category')  # Converting 'contact' column to 'category' type
df3['response'] = df3['response'].astype('category')  # Converting 'response' column to 'category' type
df3['region'] = df3['region'].astype('category')  # Converting 'region' column to 'category' type
df3['job'] = df3['job'].astype('category')  # Converting 'job' column to 'category' type
df3['marital'] = df3['marital'].astype('category')  # Converting 'marital' column to 'category' type
df3['education'] = df3['education'].astype('category')  # Converting 'education' column to 'category' type
df3['default'] = df3['default'].astype('category')  # Converting 'default' column to 'category' type
df3['housing'] = df3['housing'].astype('category')  # Converting 'housing' column to 'category' type
df3['loan'] = df3['loan'].astype('category')  # Converting 'loan' column to 'category' type
df3.info()

#create a new variable to keep the z score of duration, balance
df3['z_duration']=stats.zscore(df3['duration'])
df3['z_balance']=stats.zscore(df3['balance'])
df3.head()

#select rows with z score duration lower than -3 or higher than 3
df3[['z_duration','duration']][(df3['z_duration']<-3)|(df3['z_duration']>3)]
#select rows with z score balance lower than -3 or higher than 3
df3[['z_balance','balance']][(df3['z_balance']<-3)|(df3['z_balance']>3)]

#generate statistics about numerical varaibles
df3.describe()

#check some statistics about the duration variable
print("\n the median: ", df3['duration'].median())
print("\n the mean: ", df3['duration'].mean())
#check some statistics about the age variable
print("\n the median: ", df3['age'].median())
print("\n the mean: ", df3['age'].mean())
#check some statistics about the balance variable
print("\n the median: ", df3['balance'].median())
print("\n the mean: ", df3['balance'].mean())

# Creating a new variable 'agebin' to categorize age into different groups, while preserving the original 'age' variable.
bins = [0, 18, 35, 60, 100]  # Defining age bins
labels = ["child", "youth", "adult", "senior"]  # Assigning labels to the bins

# Creating a new column 'agebin' that stores the categorized age groups
df3['agebin'] = pd.cut(df3['age'], bins=bins, labels=labels)
print(df3)

# Crosstab of contact and response
crosstab_contact_response = pd.crosstab(df3['contact'], df3['response'])

# Create a stacked bar chart
crosstab_contact_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Contact Method vs Response")
plt.xlabel("Contact Method")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_contact_response)

# Crosstab of age and response
crosstab_agebin_response = pd.crosstab(df3['agebin'], df3['response'])

# Create a stacked bar chart
crosstab_agebin_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Age Group vs Response")
plt.xlabel("Age Group")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_agebin_response)

# Crosstab of education and response
crosstab_education_response = pd.crosstab(df3['education'], df3['response'])

# Create a stacked bar chart
crosstab_education_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Education Level vs Response")
plt.xlabel("Education Level")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_education_response)

# Crosstab of region and response
crosstab_region_response = pd.crosstab(df3['region'], df3['response'])

# Create a stacked bar chart
crosstab_region_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Region vs Response")
plt.xlabel("Region")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_region_response)

# Crosstab of job and response
crosstab_job_response = pd.crosstab(df3['job'], df3['response'])

# Create a stacked bar chart
crosstab_job_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("$")
plt.xlabel("Job Type")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_job_response)

# Crosstab of marital status and response
crosstab_marital_response = pd.crosstab(df3['marital'], df3['response'])

# Create a stacked bar chart
crosstab_marital_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Marital vs Response")
plt.xlabel("Marital")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_marital_response)

# Crosstab of credit in default and response
crosstab_default_response = pd.crosstab(df3['default'], df3['response'])

# Create a stacked bar chart
crosstab_default_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Credit in Default vs Response")
plt.xlabel("Credit in Default")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_default_response)

# Crosstab of housing laon and response
crosstab_housing_response = pd.crosstab(df3['housing'], df3['response'])

# Create a stacked bar chart
crosstab_housing_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Housing Loan vs Response")
plt.xlabel("Housing Loan")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_housing_response)

# Crosstab of personal loan and response
crosstab_loan_response = pd.crosstab(df3['loan'], df3['response'])

# Create a stacked bar chart
crosstab_loan_response.plot(kind='bar', stacked=True, figsize=(8, 6), color=['yellow', 'blue'])
plt.title("Personal Loan vs Response")
plt.xlabel("Personal Loan")
plt.ylabel("Count of Response")
plt.xticks(rotation=45)
plt.legend(title="Response", labels=["No", "Yes"])
plt.show()
stats.chi2_contingency(crosstab_loan_response)



# LR Model Start
# Converting the 'response' categorical variable into numerical codes.
df3['response'] = df3['response'].cat.codes

# Creating dummy variables for categorical features to make them suitable for machine learning models.
categorical_features = ['contact', 'region', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'agebin']

# Generate dummy variables for the categorical features in the list
dummies = pd.get_dummies(df3[categorical_features])

# Display the information about the dummy DataFrame
dummies.info()

# Display the first few rows of the dummy variables DataFrame
dummies.head()

# Extracting the target variable 'response' from the DataFrame to use it for model training.
# 'y' will be the response, that we are trying to predict.
y = df3['response']

# The following line drops the specified columns and concatenates the resulting DataFrame with the dummy variables created for categorical features.
X = pd.concat([df3.drop(categorical_features + ['response', 'custID', 'duration', 'balance'], axis=1), dummies], axis=1)

# This line removes certain features, such as 'contact_unknown', 'region_Missing', etc., that may have been included in the dummy encoding but are not needed for the model.
X = X.drop(['contact_unknown', 'region_Missing', 'job_Missing', 'marital_others', 'education_Missing', 'default_0', 'housing_0', 'loan_0', 'agebin_child'], axis=1)

# Renaming columns to remove spaces and make them more consistent for easier reference in the model
X.rename(columns={'contact_virtual assistant': 'contact_virtualassistant', 
                  'region_North East': 'region_NorthEast',
                  'region_South West': 'region_SouthWest',
                  'region_East of England': 'region_EastofEngland',
                  'region_South East': 'region_SouthEast',
                  'region_North West': 'region_NorthWest',
                  'region_West Midlands': 'region_WestMidlands',
                  'region_Yorkshire and the Humber': 'region_YorkshireandtheHumber',
                  'region_East Midlands': 'region_EastMidlands',
                  'job_domestic worker': 'job_domesticworker',
                  'job_self-employed': 'job_selfemployed'}, inplace=True)

X.info()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2222, stratify=y)

# This variable, input_variables, contains the list of features (predictors) used in the model. 
input_variables= (
    "z_duration + z_balance + age + " 
    "agebin_youth + agebin_adult + agebin_senior + "
    "contact_phone + contact_virtualassistant + "
    "region_NorthEast + region_SouthWest + region_EastofEngland + region_SouthEast + region_NorthWest + region_WestMidlands + region_YorkshireandtheHumber + region_EastMidlands + region_London + "
    "job_admin + job_domesticworker + job_entrepreneur + job_management + job_others + job_retired + job_selfemployed + job_services + job_student + job_technician + "
    "marital_married + marital_single + "
    "education_primary + education_secondary + education_tertiary + "
    "default_1 + housing_1 + loan_1"
    )                                                 

# Concatenate the response variable with input variables to create the regression formula
the_formula = "response ~ " + input_variables
print("Formula: ", the_formula, "\n")  

trainingdata = pd.concat([X_train, y_train], axis=1)  # Combine the features and target variable for training data
trainingdata.info()  # Display the structure and summary of the training data
trainingdata.head()  # Show the first few rows of the training data

LR_Model = smf.logit(the_formula, data=trainingdata).fit()  # Fit a logistic regression model using the formula
LR_Model.summary()  # Display the summary of the logistic regression model to examine coefficients and statistics

#input_variable P < 0.05                                          
input_variables= (
    "z_duration + z_balance + age + " 
    "agebin_senior + "
    "contact_phone + contact_virtualassistant + "
    "job_domesticworker + job_entrepreneur + job_others + job_selfemployed + job_student + job_technician + "
    "marital_married + marital_single + "
    "education_primary + education_secondary + "
    "housing_1 + loan_1"
    )    

# Concatenate the response variable with input variables to create the regression formula
the_formula = "response ~ " + input_variables
print("Formula: ", the_formula, "\n")  

LR_Model = smf.logit(the_formula, data=trainingdata).fit()  # Fit a logistic regression model using the formula
LR_Model.summary()  # Display the summary of the logistic regression model to examine coefficients and statistics

#input_variable P < 0.05  
input_variables= (
    "z_duration + z_balance + age + " 
    "agebin_senior + "
    "contact_phone + contact_virtualassistant + "
    "job_domesticworker + job_entrepreneur + job_others + job_selfemployed + job_student + job_technician + "
    "marital_single + "
    "education_primary + education_secondary + "
    "housing_1 + loan_1"
    )                                               

# Concatenate the response variable with input variables to create the regression formula
the_formula= "response ~ " + input_variables
print("Formula: ", the_formula, "\n")

LR_Model = smf.logit(the_formula, data=trainingdata).fit()  # Fit a logistic regression model using the formula
LR_Model.summary()  # Display the summary of the logistic regression model to examine coefficients and statistics

# Predict probabilities on the test set using the logistic regression model
pred_prob_X_test_LR = LR_Model.predict(X_test)

# Adjust the cut-off parameter to 0.3 to determine the classification threshold
pred_X_test_class_LR = (pred_prob_X_test_LR > 0.3).astype(int)

# Calculate confusion matrix and performance metrics
print("Confusion matrix: \n", metrics.confusion_matrix(y_test, pred_X_test_class_LR))

# Compute performance metrics
Accuracy = metrics.accuracy_score(y_test, pred_X_test_class_LR)
Sensitivity_recall = metrics.recall_score(y_test, pred_X_test_class_LR)
Specificity = metrics.recall_score(y_test, pred_X_test_class_LR, pos_label=0)
F1_Score = metrics.f1_score(y_test, pred_X_test_class_LR)

# Print results
print({"LR: Accuracy": Accuracy,
       "Sensitivity/recall": Sensitivity_recall,
       "Specificity": Specificity,
       "F1-Score": F1_Score})

# Display confusion matrix with the adjusted threshold (0.3)
cm = metrics.confusion_matrix(y_test, pred_X_test_class_LR)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-response', 'Response'])
cm_display.plot()
plt.title("Logistic Regression Confusion Matrix (Cut-Off 0.3)")

# Function to plot the ROC curve and calculate AUC
def plot_roc_curve(true_y, y_prob):
    """
    Plots the ROC curve and calculates the AUC (Area Under the Curve).
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_y, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot ROC curve
    plt.title('Receiver Operating Characteristic_LG')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('(1-Specificity)/False Positive Rate')
    plt.ylabel('Sensitivity/True Positive Rate')
    
# Call the function to plot ROC curve for the Logistic Regression model
plot_roc_curve(y_test, pred_prob_X_test_LR)

# Creating a DataFrame for predicted probabilities
LR_prob = pd.DataFrame({"Non-Response": 1 - pred_prob_X_test_LR, "Response": pred_prob_X_test_LR})

# Plotting the Cumulative Gain Curve for Logistic Regression
skplt.metrics.plot_cumulative_gain(y_test, LR_prob)
plt.title("Cumulative Gain Curve: Logistic Regression")
plt.show()




#Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Extracting the target variable 'response' from the DataFrame to use it for model training.
# 'y' will be the response, that we are trying to predict.
y = df3['response']

# Concatenating the features after dropping unnecessary ones and adding dummy variables
X = pd.concat([df3.drop(categorical_features + ['response', 'custID', 'duration', 'balance'], axis=1), dummies], axis=1)

# Displaying information about the resulting DataFrame X, including column types and non-null counts
X.info()

# Renaming columns for consistency and better readability
X.rename(columns={'contact_virtual assistant': 'contact_virtualassistant', 
                  'region_North East': 'region_NorthEast',
                  'region_South West': 'region_SouthWest',
                  'region_East of England': 'region_EastofEngland',
                  'region_South East': 'region_SouthEast',
                  'region_North West': 'region_NorthWest',
                  'region_West Midlands': 'region_WestMidlands',
                  'region_Yorkshire and the Humber': 'region_YorkshireandtheHumber',
                  'region_East Midlands': 'region_EastMidlands',
                  'job_domestic worker': 'job_domesticworker',
                  'job_self-employed': 'job_selfemployed'}, inplace=True)

# Displaying updated DataFrame information
X.info()

# Splitting the data into training and testing sets
# This ensures that the data is split into 70% for training and 30% for testing, 
# while maintaining the same proportion of target variable 'y' in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2222, stratify=y)

# Initializing and training a Decision Tree classifier with specific hyperparameters
clf = DecisionTreeClassifier(max_depth=8, min_samples_split=100, random_state=2222, criterion='gini')

# Training the Decision Tree model on the training data (X_train, y_train)
# This model is set with a maximum depth of 8 to avoid overfitting and a minimum of 100 samples required to split a node,
# using the Gini impurity criterion for measuring the quality of splits.
Tree_Model = clf.fit(X_train, y_train)

# Predicting the probability of the positive class (class 1) for each sample in the test set
pred_prob_X_test_Tree = Tree_Model.predict_proba(X_test)[:, 1]

# Setting the cutoff threshold to 0.3 for converting predicted probabilities to binary class labels
cutoff = 0.3
y_pred_tree = (pred_prob_X_test_Tree >= cutoff).astype(int)

cm = metrics.confusion_matrix(y_test, y_pred_tree)
print("Confusion Matrix:\n", cm)

# Plotting the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-response','Response'])
cm_display.plot()
plt.title("Decision Tree Confusion Matrix (Cut-Off 0.3)")

# Calculate and print Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred_tree)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate and print Sensitivity (Recall for Response)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
print("Sensitivity: {:.2f}%".format(sensitivity * 100))

# Calculate and print Specificity (True Negative Rate)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print("Specificity: {:.2f}%".format(specificity * 100))

# Calculate F1 Score
F1_Score = metrics.f1_score(y_test, y_pred_tree)
print("F1 Score: {:.2f}".format(F1_Score))

# Creating a list of feature names and class labels
feature_names = X_train.columns.tolist()
class_names = ["not response", "response"]

# Visualizing the decision tree using plot_tree
plt.figure(figsize=(12, 8)) 
plot_tree(Tree_Model, feature_names=feature_names, filled=True, class_names=class_names, fontsize=10)
plt.show()

# Creating a DataFrame to display feature importances
dfFeatures = pd.DataFrame({'Features': feature_names, 'Importances': Tree_Model.feature_importances_})

# Sorting the features by importance and displaying the top 10
dfFeatures.sort_values(by='Importances', ascending=False).head(10)

# Plotting the ROC curve for the Decision Tree model
plot_roc_curve(y_test, pred_prob_X_test_Tree)
plt.title('Receiver Operating Characteristic_DT')

# Creating a DataFrame with predicted probabilities for non-response and response
DT_prob = pd.DataFrame(pred_prob_X_test_Tree)
DT_prob = pd.DataFrame({"Non-Response": 1 - pred_prob_X_test_Tree, "Response": pred_prob_X_test_Tree})

# Plotting the Cumulative Gain Curve for the Decision Tree model
skplt.metrics.plot_cumulative_gain(y_test, DT_prob)
plt.title("Cumulative Gain Curve: Decision Tree")
plt.show()





#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Training a Random Forest model on the training data
rf_model_orig = RandomForestClassifier(random_state=2222)
rf_model_orig.fit(X_train, y_train)

# Predicting the probabilities of the test set using the trained Random Forest model
pred_prob_X_test_RForest = rf_model_orig.predict_proba(X_test)[:, 1]

# Set cutoff value for classification
cutoff = 0.3

# Convert predicted probabilities to binary labels
y_pred_forest = (pred_prob_X_test_RForest >= cutoff).astype(int)

# Confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred_forest)
print("Confusion Matrix:\n", cm)

# Calculate evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred_forest)
sensitivity = metrics.recall_score(y_test, y_pred_forest)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity (True Negative Rate)
F1_Score = metrics.f1_score(y_test, y_pred_forest)

# Print results
print({"Random Forest: Accuracy": accuracy,
       "Sensitivity/Recall": sensitivity,
       "Specificity": specificity,
       "F1-Score": F1_Score})

# Plot confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-response', 'Response'])
cm_display.plot()
plt.title("Random Forest Confusion Matrix (Cut-Off 0.3)")
plt.show()

# Plot the ROC curve for Random Forest model
plot_roc_curve(y_test, pred_prob_X_test_RForest)
plt.title('Receiver Operating Characteristic_RF')
plt.show()

# Create a DataFrame with probabilities for both classes (Non-Response and Response)
RF_prob = pd.DataFrame({"Non-Response": 1 - pred_prob_X_test_RForest, "Response": pred_prob_X_test_RForest})

# Plot the Cumulative Gain chart for the Random Forest model
skplt.metrics.plot_cumulative_gain(y_test, RF_prob)
plt.title("Cumulative Gain Chart: Random Forest")
plt.show()

# Ensure important_features matches the features used in the model
important_features = X_train.columns  # Use the correct feature names from your training dataset

# Extract feature importances
final_importances = rf_model_orig.feature_importances_

# Create a DataFrame with features and their importance
final_features = pd.DataFrame({
    'Feature': important_features,
    'Importance': final_importances
})

# Sort the DataFrame by importance in descending order
final_features = final_features.sort_values(by='Importance', ascending=False)

# Display the top features
print(final_features.head())

# Plot a horizontal bar chart to visualize the feature importances in Random Forest
plt.figure(figsize=(10, 12))
plt.barh(final_features['Feature'], final_features['Importance'], color='skyblue')  
plt.xlabel('Importance')  
plt.ylabel('Feature')  
plt.title('Feature Importances in Random Forest')  # Chart title
plt.gca().invert_yaxis()  
plt.show()  

