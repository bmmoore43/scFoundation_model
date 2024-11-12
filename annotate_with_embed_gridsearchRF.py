# load packages and set working directory
import scanpy as sc
import pandas as pd
import pickle
import numpy as np
import os
import anndata
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

os. getcwd()
os. chdir('/w5home/bmoore/scFoundation/annotation/')
os. getcwd()

# load ann data to get labels
adata = sc.read_h5ad("/w5home/bmoore/scRNAseq/GAMM/GAMM_S2/output_20230830_155530/GAMM_S2_clabeled-clusters_0.5.h5ad")
print(len(adata.obs))

# Load the embeddings from the .npy file
embeddings = np.load('../Gamm_cell_embed/Gamm_01B-resolution_singlecell_cell_embedding_a5_resolution_lognorm.npy')

# Extract the labels
labels = adata.obs['CellType_manual'].values

# Check if the number of embeddings matches the number of labels
assert embeddings.shape[0] == labels.shape[0], "Number of embeddings does not match number of labels."

# Combine the embeddings and labels into a dictionary
data = {
    'embeddings': embeddings,
    'labels': labels
}

# Save the combined data as a pickle file
with open('Gamm_01B-resolution_singlecell_cell_embedding_a5_resolution_lognorm_withlabels.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Embeddings and labels have been successfully combined and saved as a pickle file.")

# Run with cross validation grid search- this takes a long time!!

# Assuming 'data' is a dictionary with 'embeddings' and 'labels'
embeddings = data['embeddings']  # Shape: (n_samples, n_features)
labels = data['labels']  # Shape: (n_samples,)
indices = list(range(len(labels)))
indices = np.array(indices)

# remove unknowns
mask = labels != 'unknown'
embeddings_filtered = embeddings[mask]
labels_filtered = labels[mask]
indices_filtered = indices[mask]

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30, None]
}

# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

print("Starting GridSearchCV...")
# Initialize GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=16) # n_jobs sets # of cpus
print("GridSearchCV initialized. Fitting GridSearch model...")
# Fit GridSearchCV
grid_search.fit(embeddings_filtered, labels_filtered)

print("Finding best model...")
# Get the best model
best_model = grid_search.best_estimator_

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# Proceed with training the best model on the entire dataset (optional)
#best_model.fit(embeddings_filtered, labels_filtered)

print("Splitting data into training and testing sets...")
# If you want to split into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    embeddings_filtered, labels_filtered, indices_filtered, test_size=0.2, random_state=42
)

print("Training the best model on the training set...")
# Train the best model on the training set
best_model.fit(X_train, y_train)

print("Making predictions on the test set...")
# Make predictions on the test set
y_pred = best_model.predict(X_test)

print("Evaluating the classifier...")
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report dictionary to a pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

print(f'Accuracy: {accuracy}')
print(f'Classification Report DataFrame:\n{report_df}')

# Save the classification report to a CSV file if needed
report_df.to_csv('classification_report_RFwithCV.csv')

print("Adding predictions to ann data object...")
# Check if the number of test embeddings matches the number of predictions
assert len(y_pred) == X_test.shape[0], "Number of predictions does not match the number of test embeddings."

# Ensure the length of test_indices matches the length of y_pred
assert len(test_indices) == len(y_pred), "The number of test indices does not match the number of predictions."

# # subset test data
print(adata.obs.index[test_indices])
adata_subset = adata[adata.obs.index[test_indices],]

print("Subsetted data: ",adata_subset)
# # add predicted data to ann object
adata_subset.obs["scfound_pred_withCV"] = pd.Categorical(y_pred)

# Visulaize the results
print("Visualizing the results...")

import seaborn as sns
sns.set_style("whitegrid")
# visualize predictions
import matplotlib.pyplot as plt
import random
import matplotlib

# view subset of data with predictions
plt.rcParams['axes.facecolor']='white'
plt.figure(figsize=(6,5)) 
with plt.rc_context():
    plt.rcParams.update({'font.size': 12})
    sc.pl.embedding(
        adata_subset,
        basis="umap",
        frameon=False,
        color=["CellType_manual", "scfound_pred_withCV"], 
        ncols=2
    )
    plt.savefig("scfound_pred_withCV.pdf")