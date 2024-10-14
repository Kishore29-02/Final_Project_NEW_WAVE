# %%
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import pandas as pd
%pip install pandas scikit-learn imblearn matplotlib plotly nbformat seaborn

# %% [markdown]
# # Data Loading

# %%

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
df.info()

# %%
df.isnull().sum()

# %%
df.interpolate(inplace=True)

# %%
df.isnull().sum()

# %%
df.head()

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
