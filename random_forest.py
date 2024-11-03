# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature Engineering for `final_df.pkl`
#
# In this notebook, we preprocess the dataset `final_df.pkl` for training machine learning models on network intrusion malware classification. We'll perform feature engineering, handle high-cardinality categorical variables, and prepare the data for model training.

# %%
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_pickle('final_df.pkl')
print(f"Data size: {df.size}")
print(f"Data types before processing:\n{df.dtypes}")

# Parse 'frame.time' as datetime and set it as index
df['frame.time'] = pd.to_datetime(df['frame.time'].str.strip(), format='%Y %H:%M:%S.%f')
df.set_index('frame.time', inplace=True)

# %% [markdown]
# ## Drop Constant Columns

# %%
# Drop columns with only one unique value
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
df.drop(columns=constant_columns, inplace=True)
print(f"Dropped constant columns: {constant_columns}")

# additional cols 
cols_to_drop = ['http.request.full_uri']# , 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4', 'arp.src.proto_ipv4' , 'tcp.srcport', 'tcp.dstport']
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# %% [markdown]
# ## Drop Unnecessary Columns

# %%
# Drop raw byte columns that are not useful for analysis

# Replace '0' with empty strings in specific columns and calculate string lengths
df['http.file_data'] = df['http.file_data'].replace('0', '').astype(str)
df['http.request.uri.query'] = df['http.request.uri.query'].replace('0', '').astype(str)
df['http.request.version'] = df['http.request.version'].replace('0', '').astype(str)
df['mqtt.msg'] = df['mqtt.msg'].replace('0', '').astype(str)
df['tcp.payload'] = df['tcp.payload'].replace('0', '').astype(str)
df['tcp.options'] = df['tcp.options'].replace('0', '').astype(str)

# Convert columns to length of each string as numerical values
df['http.file_data'] = df['http.file_data'].apply(len)
df['http.request.uri.query'] = df['http.request.uri.query'].apply(len)
df['http.request.version'] = df['http.request.version'].apply(len)
df['mqtt.msg'] = df['mqtt.msg'].apply(len)
df['tcp.payload'] = df['tcp.payload'].apply(len)
df['tcp.options'] = df['tcp.options'].apply(len)

# raw_byte_columns = ['tcp.payload']
# df.drop(columns=raw_byte_columns, inplace=True)
# print(f"Dropped raw byte columns: {raw_byte_columns}")

# Drop TCP flags column (assuming flags have been processed or are not needed)
flag_columns = ['tcp.flags']
df.drop(columns=flag_columns, inplace=True)
print(f"Dropped original flag columns: {flag_columns}")

# %% [markdown]
# ## Remove 'Attack_type' Column to Prevent Data Leakage

# %%
if 'Attack_label' in df.columns:
    df.drop(columns=['Attack_label'], inplace=True)

# %% [markdown]
# ## Convert Columns to Numeric

# %%
# Convert specific columns to numeric where applicable
convert_to_num_columns = ['arp.opcode', 'icmp.checksum', 'tcp.checksum']
df[convert_to_num_columns] = df[convert_to_num_columns].apply(pd.to_numeric, errors='coerce')

# %% [markdown]
# ## Replace Non-Numeric Zero Values in Object Columns

# %%
# Replace '0.0' and '0' with '0' in object type columns
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].replace({'0.0': '0', '0': '0'})

# %% [markdown]
# ## Parse String Numbers and Hex Values in MQTT Fields

# %%
# Define columns to parse
cols_to_parse = ['mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags']

# Function to parse hexadecimal and string numbers
def parse_string_and_hex_columns(df, columns):
    for column in columns:
        df[column] = df[column].apply(
            lambda value: int(value, 16) if isinstance(value, str) and value.startswith('0x')
            else int(value) if isinstance(value, str) and value.isdigit()
            else pd.NA
        )
    return df

# Apply the function
df = parse_string_and_hex_columns(df, cols_to_parse)
print(f"Parsed columns: {cols_to_parse}")

# %% [markdown]
# ## Handle High-Cardinality Categorical Variables

# %%
# Identify categorical columns
categorical_columns = [
    'ip.src_host', 'ip.dst_host',
    'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
    'tcp.srcport', 'tcp.dstport',
    'http.request.method', 'http.referer',
    # 'http.request.full_uri',
    'http.response',
    'mqtt.protoname', 'mqtt.topic'
]

# Convert categorical columns to strings
df[categorical_columns] = df[categorical_columns].astype(str)

# Analyze cardinality
for col in categorical_columns:
    print(f"Column '{col}' has {df[col].nunique()} unique values.")

# %% [markdown]
# ### Apply Target Encoding to High-Cardinality Columns

# %%
# High-cardinality columns (more than 100 unique values)
high_cardinality_cols = [col for col in categorical_columns if df[col].nunique() > 100]
print(f"High-cardinality columns: {high_cardinality_cols}")

# Low-cardinality columns
low_cardinality_cols = list(set(categorical_columns) - set(high_cardinality_cols))
print(f"Low-cardinality columns: {low_cardinality_cols}")

# Target variable for encoding
target = 'Attack_type'

# Remove 'Attack_label' from categorical columns if present
if target in high_cardinality_cols:
    high_cardinality_cols.remove(target)
if target in low_cardinality_cols:
    low_cardinality_cols.remove(target)

# Split data (using stratify to maintain class balance)
X = df.drop(columns=[target])
y = df[target]

# Encode the string class labels into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply Target Encoding to high-cardinality columns on training data only
te = TargetEncoder(cols=high_cardinality_cols)
X_train[high_cardinality_cols] = te.fit_transform(X_train[high_cardinality_cols], y_train)
X_test[high_cardinality_cols] = te.transform(X_test[high_cardinality_cols])

# %% [markdown]
# ### One-Hot Encode Low-Cardinality Columns

# %%
# One-Hot Encode low-cardinality categorical columns
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

# Fit encoder on training data
onehot_encoder.fit(X_train[low_cardinality_cols])

# Transform both training and test data
X_train_encoded = onehot_encoder.transform(X_train[low_cardinality_cols])
X_test_encoded = onehot_encoder.transform(X_test[low_cardinality_cols])

# Create DataFrames from the encoded data
encoded_columns = onehot_encoder.get_feature_names_out(low_cardinality_cols)
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

# Drop original low-cardinality columns and join encoded columns
X_train_encoded_df = X_train_encoded_df.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
X_test_encoded_df = X_test_encoded_df.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train = X_train.drop(columns=low_cardinality_cols).join(X_train_encoded_df)
X_test = X_test.drop(columns=low_cardinality_cols).join(X_test_encoded_df)

# %% [markdown]
# ## Handle Missing Values

# %%
# Fill missing values in training and test sets separately
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# %% [markdown]
# ## Feature Scaling

# %%
# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Train a Machine Learning Model

# %%
# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Evaluate the Model

# %%
# Predict on test data
y_pred = rf_model.predict(X_test_scaled)

# Print classification report with proper class names
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Print confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# Plot confusion matrix with proper class names
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ## Feature Importance

# %%
# Get feature importances from the model
importances = rf_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()
