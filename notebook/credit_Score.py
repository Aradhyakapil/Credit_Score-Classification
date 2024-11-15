# %%
import pandas as pd
import numpy as np
!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv('../../data/train.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
df.isna().sum()

# %%
df1 = df.drop(["ID","Customer_ID","Month","Name","Occupation","Type_of_Loan","Payment_of_Min_Amount","Total_EMI_per_month","Amount_invested_monthly","SSN"],axis=1)

# %%
df1 = df1.dropna()

# %%
df1.info()

# %%
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df[~df["Age"].apply(is_float)]

# %%
def convert_to_Float(x):
    age = list(x.split("_"))
    return float(age[0])

df1["Age"] =df1["Age"].apply(convert_to_Float)

# %%
df1 = df1[df1["Age"] >= 10]
df1 = df1[df1["Age"] <= 100]

# %%
df1.describe()

# %%
df1.info()

# %%
df1[~df1["Annual_Income"].apply(is_float)]

# %%
df1["Annual_Income"] =df1["Annual_Income"].apply(convert_to_Float)

# %%
df1.describe()

# %%
bins = [df1['Annual_Income'].quantile(0), df1['Annual_Income'].quantile(0.2), df1['Annual_Income'].quantile(0.4),
        df1['Annual_Income'].quantile(0.6), df1['Annual_Income'].quantile(0.8), df1['Annual_Income'].quantile(1)]

labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

df1['Salary_Range'] = pd.cut(df1['Annual_Income'], bins=bins, labels=labels)

df1

# %%
pivot_table = pd.pivot_table(df1, values='Annual_Income', index='Salary_Range', columns='Credit_Score', aggfunc='count', fill_value=0)

print(pivot_table)
#Plot a bar graph
pivot_table.plot(kind='bar', stacked=True)
plt.title('Credit Score Distribution by Salary Range')
plt.xlabel('Salary Range')
plt.ylabel('Count')
plt.legend(title='Credit Score')
plt.show()

# %%
df2 = df1[df1["Annual_Income"] < 12*df1["Monthly_Inhand_Salary"]]

# %%
df2.groupby("Credit_Score")["Credit_Score"].agg("count")

# %%
df1["Num_Bank_Accounts"] = df1["Num_Bank_Accounts"].apply(lambda x: abs(x) if x !=0 else x+1)

# %%
df1.describe()

# %%
# df2 = df1

# %%
# df1.drop(["Annual_Income","Monthly_Inhand_Salary"],axis=1,inplace=True)

# %%
df1.info()

# %%
df1["Num_of_Loan"] =df1["Num_of_Loan"].apply(convert_to_Float)
df1["Num_of_Delayed_Payment"] = df1["Num_of_Delayed_Payment"].apply(convert_to_Float)

# %%
df1["Changed_Credit_Limit"].replace("_","0",inplace=True)
df1["Changed_Credit_Limit"] = df1["Changed_Credit_Limit"].apply(convert_to_Float)

# %%
df1["Credit_Mix"].replace("_","Standard",inplace=True)
credit_mix = {"Good":0,"Standard":1,"Bad":2}
df1["Credit_Mix"].replace(credit_mix,inplace=True)

# %%
df1["Outstanding_Debt"] = df1["Outstanding_Debt"].apply(convert_to_Float)

# %%
def get_year(x):
    x = x.split()
    return int(x[0])
def get_month(x):
    x = x.split()
    return int(x[3])

df1["Credit_History_Year"] = df1["Credit_History_Age"].apply(get_year)
df1["Credit_History_Month"] = df1["Credit_History_Age"].apply(get_month)


# %%
df1 = df1.drop("Credit_History_Age",axis=1)

# %%
credit_score = {"Good":0,"Standard":1,"Poor":2}
df1["Credit_Score"].replace(credit_score,inplace=True)

# %%
df1 = df1[df1["Payment_Behaviour"]!="!@9#%8"]

# %%
df1["Payment_Behaviour"].unique()

# %%
payment_behavior_dict = {
    'High_spent_Small_value_payments': 0,
    'Low_spent_Small_value_payments': 1,
    'High_spent_Large_value_payments': 2,
    'Low_spent_Large_value_payments': 3,
    'High_spent_Medium_value_payments': 4,
    'Low_spent_Medium_value_payments': 5
}

df1["Payment_Behaviour"].replace(payment_behavior_dict,inplace=True)

# %%
df1["Salary_Range"]

# %%
df1.isna().sum()

# %%
df1 = df1.dropna()

# %%
salary_range_dict = {
    'Very Low': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}
df1["Salary_Range"].replace(salary_range_dict,inplace=True)

# %%
df1 = df1[df1["Monthly_Balance"].apply(is_float)]
# df1["Monthly_Balance"].apply(convert_to_Float)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' containing the numeric columns
# Extract the numeric columns you want to include in the heatmap
# Calculate the correlation matrix
correlation_matrix = df1.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()


# %%
from sklearn.model_selection import train_test_split

X = df1.drop("Credit_Score",axis=1)
y = df1["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.30, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# %%
rf.score(X_test,y_test)

# %%
df1.columns

# %%
X = df1[["Credit_History_Year", "Salary_Range", "Outstanding_Debt", "Credit_Mix", "Changed_Credit_Limit", "Delay_from_due_date","Age"]]
y = df1["Credit_Score"]

# %%
df1.dropna(inplace=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.30, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# %%
rf.score(X_test,y_test)

# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

predictions = rf.predict(X_test)
print('Classification Report', classification_report(y_test, predictions))
print('\n')
print('Confusion Matrix', confusion_matrix(y_test, predictions))
print('\n')
print('Accuracy Score', accuracy_score(y_test, predictions))

# %%
df1.columns

# %%
X = df1[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", 
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt", "Credit_History_Year", "Monthly_Balance"]]
y = df1['Credit_Score']



# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
model = RandomForestClassifier(n_estimators=30, class_weight='balanced')

# %%
model.fit(X_train, y_train)

# %%
# Making predictions on our model using the test data
predictions = model.predict(X_test)

# %%
print('Classification Report', classification_report(y_test, predictions))
print('\n')
print('Confusion Matrix', confusion_matrix(y_test, predictions))
print('\n')
print('Accuracy Score', accuracy_score(y_test, predictions))

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')  # Use 'weighted' average for multi-class
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Data for the bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy, precision, recall, f1]

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(metrics, scores)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Detailed Model Performance Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')

# Display the chart
plt.show()

# %%

# %pip install imbalanced-learn

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy="minority")
X_sm, y_sm = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.33, random_state=42, stratify=y_sm)

# %%
model = RandomForestClassifier(n_estimators=50, class_weight='balanced')
model.fit(X_train, y_train)
# Making predictions on our model using the test data
predictions = model.predict(X_test)

print('Classification Report', classification_report(y_test, predictions))
print('\n')
print('Confusion Matrix', confusion_matrix(y_test, predictions))
print('\n')
print('Accuracy Score', accuracy_score(y_test, predictions))


# %%
# Plot the Confusion Matrix using Seaborn
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good', 'Standard', 'Poor'],
            yticklabels=['Good', 'Standard', 'Poor'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
import joblib

joblib.dump(model, "../../model/model_v2.joblib")

# %%
df2 = df1[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", 
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt", "Credit_History_Year", "Monthly_Balance"]]

# %%
df2.to_csv("../../data/new.csv")

# %%
df2.info()


