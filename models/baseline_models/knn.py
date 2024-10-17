# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline model - KNN 
#
# In this notebook, we aim to get a gauge of how well a model such as KNN will do in classifying the different attack types using our data in EDA_df.csv, which contains 200k rows sampled from our main data

# %% [markdown]
# ## Read the data from csv

# %%
import pandas as pd

data = pd.read_csv('../../data/EDA_df.csv')

# %% [markdown]
# ## Preparing the data for training

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
                             
X = data.drop(columns=['Attack_type','Attack_label']) 
X = X.sample(n=20000, random_state=47)
y = data['Attack_type']
y = y.sample(n=20000, random_state=47)

X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the columns to ensure fairness across attributes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Prepare the KNN classifier

# %%
from sklearn.neighbors import KNeighborsClassifier
import joblib

knn = KNeighborsClassifier(n_neighbors=10)  # You can change n_neighbors

# Fit the model to the training data
knn.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(knn, '../../trained_model/knn_model.pkl')

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

knn = joblib.load('../../trained_model/knn_model.pkl')
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Precision: {precision:.3f}')

recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Recall: {recall:.3f}')

f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'F1-Score: {f1:.3f}')



# %%
