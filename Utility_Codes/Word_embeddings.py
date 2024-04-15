import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
# pass the data
data = pd.read_csv("/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/Data/FDA/FDA_Nuts.csv")
# Assuming data is your DataFrame containing both categorical and numerical data

# separate Categorical and Numerical Data
categorical_data = data.select_dtypes(include=['object'])
numerical_data = data.select_dtypes(exclude=['object'])

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to tokenize and obtain embeddings for a single value
def get_embeddings(value):
    inputs = tokenizer(value, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embeddings

# Function to obtain embeddings for each value in a categorical column
def get_column_embeddings(column):
    embeddings_list = []
    for value in column:
        embeddings = get_embeddings(value)
        embeddings_list.append(embeddings)
    return embeddings_list

# Dictionary to store aggregated embeddings for each categorical column
categorical_embeddings_aggregated = {}

# Generate Word Embeddings for Categorical Data
for col in categorical_data.columns:
    print(f'Obtaining embeddings for {col}...')
    # Obtain embeddings for values in the column
    embeddings = get_column_embeddings(categorical_data[col])
    # Aggregate embeddings (e.g., average pooling)
    aggregated_embeddings = np.mean(embeddings, axis=0)
    # Store aggregated embeddings in the dictionary
    categorical_embeddings_aggregated[col] = aggregated_embeddings

# Convert categorical_embeddings_aggregated dictionary to DataFrame
categorical_embeddings_df = pd.DataFrame(categorical_embeddings_aggregated)




## Combine Numerical and Categorical Data with Embeddings
data_with_embeddings = pd.concat([numerical_data.reset_index(drop=True), categorical_embeddings_df.reset_index(drop=True)], axis=1)
