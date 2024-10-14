# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# %%
# import plotly.express as px

# %% [markdown]
# # Data Loading

# %%
test = pd.read_csv("./Data/test.csv")
train = pd.read_csv("./Data/train.csv")

# %%
train['education'] = train['education'].fillna(train.education.mode()[0])
test['education'] = test['education'].fillna(test.education.mode()[0])

# fill the missing values in previous_year_rating
train['previous_year_rating'] = train['previous_year_rating'].fillna(0)
test['previous_year_rating'] = test['previous_year_rating'].fillna(0)

# %%
df = pd.concat([train, test], axis=0)
# df = train.copy()
df.size

# %%
df = df.drop_duplicates()


# %%
df.interpolate(inplace=True)

# %%
df.isnull().sum()

# %% [markdown]
# # EXPLORATORY DATA ANALYSIS

# %% [markdown]
# ### No of Unique Values

# %%
unique_values = train.select_dtypes(include='number').nunique()
plt.figure(figsize=(10, 6))
sns.barplot(x=unique_values.index, y=unique_values.values)
plt.title('Number of Unique Values')
plt.xlabel('Features')
plt.ylabel('Number of Unique Values')
plt.yscale('log')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Education status - Awards won

# %%
plt.figure(figsize=(8, 8))
education_counts = df['education'].value_counts()
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%',
        startangle=140, colors=['blue', 'red', 'green', 'orange'])
plt.title('Distribution of Education Status')
plt.axis('equal')
plt.show()

# %% [markdown]
# ### Number of employess and their education status - Promotion

# %%
plt.figure(figsize=(6, 6))
sns.barplot(x='education', y='employee_id', hue='is_promoted', data=df)
plt.show()

# %% [markdown]
# ### Gender Distribution

# %%
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df, palette=['blue', 'red'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Length of Service by Recruitment Channel

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='recruitment_channel', y='length_of_service',
            data=train, palette='Set1')
plt.title('Length of Service by Recruitment Channel')
plt.xlabel('Recruitment Channel')
plt.ylabel('Length of Service')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Age Distribution by Education

# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='age', hue='education',
             multiple='stack', palette='Set1', bins=30)
plt.title('Age Distribution by Education')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ### Correlation of features

# %%
# Create a new variable 'df_encoded' to store the modified DataFrame
df_encoded = df.copy()

# Drop 'employee_id' column
# df_encoded = df_encoded.drop(['employee_id'], axis=1)

# Columns that are categorical
categorical_cols = ['department', 'education',
                    'gender', 'recruitment_channel', 'region']

# Encode the categorical columns using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Now 'df_encoded' is ready for correlation analysis
correlation_matrix = df_encoded.corr()

# Plotting the correlation matrix using Seaborn heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Features')
plt.show()


# %%
# Create a new variable 'df_onehot' to store the modified DataFrame
df_onehot = df.copy()

# Drop 'employee_id' column
df_onehot = df_onehot.drop(['employee_id'], axis=1)

# Columns that are categorical
categorical_cols = ['department', 'education',
                    'gender', 'recruitment_channel', 'region']

# Apply one-hot encoding to categorical columns
df_onehot = pd.get_dummies(
    df_onehot, columns=categorical_cols, drop_first=True)

# Find the correlation matrix
correlation_matrix = df_onehot.corr()

# Extract correlation with 'is_promoted'
correlation_with_promoted = correlation_matrix['is_promoted'].sort_values(
    ascending=False)

# Print the correlations with 'is_promoted'
print(correlation_with_promoted)


# %% [markdown]
# ### Creating a Metric of Sum

# %%
df['sum_metric'] = df['awards_won?'] + \
    df['KPIs_met >80%'] + df['previous_year_rating']
df.head()

# %% [markdown]
# ### Creating a total score column

# %%
df['total_score'] = df['avg_training_score'] * df['no_of_trainings']
df.head()

# %% [markdown]
# ### Drop 'recruitment_channel', 'region', 'employee_id' columns

# %%
df = df.drop(['recruitment_channel', 'region', 'employee_id'], axis=1)
df.head()

# %%
df = df.drop(df[(df['KPIs_met >80%'] == 0) & (df['previous_year_rating'] == 1.0) &
                (df['awards_won?'] == 0) & (df['avg_training_score'] < 60)].index)
df.head()

# %% [markdown]
# ### Encoding Categorical Data

# %%

df['education'] = df['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                          (3, 2, 1))
# test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
#   (3, 2, 1))

le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])
# test['department'] = le.fit_transform(test['department'])
df['gender'] = le.fit_transform(df['gender'])
# test['gender'] = le.fit_transform(test['gender'])

df.head()

# %% [markdown]
# ### Split the data to training and testing set

# %%
# Define the target variable (y) and features (X)
# Features (all columns except 'is_promoted')
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Display the size of the training and testing sets
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


# %% [markdown]
# # Model Training

# %% [markdown]
# ### Decision Tree

# %%
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# %% [markdown]
# ### Random forest

# %%
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# %% [markdown]
# ### Logistic Regression Model

# %%
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# %% [markdown]
# # Model Evaluation

# %%


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
                                   'Not Promoted', 'Promoted'])
    confusion = confusion_matrix(y_test, y_pred)

    # Print the results
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(confusion)
    # print("\n" + "="*50 + "\n")

    return accuracy


# %% [markdown]
# ### Decision tree

# %%
accuracy_dt = evaluate_model(
    dt_model, X_train, y_train, X_test, y_test, "Decision Tree")

# %% [markdown]
# ### Random Forest

# %%
accuracy_rf = evaluate_model(
    rf_model, X_train, y_train, X_test, y_test, "Random Forest")

# %% [markdown]
# ### Logistic Regression

# %%
accuracy_lr = evaluate_model(
    lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")

# %% [markdown]
# ### Model Accuracy Comparision

# %%

model_results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [accuracy_dt * 100, accuracy_rf * 100, accuracy_lr * 100]
})

print(model_results)
