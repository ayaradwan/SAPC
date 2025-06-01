import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
import string
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score,confusion_matrix, davies_bouldin_score, accuracy_score, classification_report, \
    mean_squared_error
from geneticalgorithm import geneticalgorithm as ga  # Genetic Algorithm Library

import warnings
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load the CSV file
df = pd.read_csv("PSOREQ/3.Smart_ranked_requirements_g10.csv")


# Load stopwords
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation

# Define preprocessing function to extract content words only
def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = []

    for token in doc:
        if (token.text not in stop_words and
            token.text not in punctuations and
            token.is_alpha and
            token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]):  # Keep content words
            cleaned_tokens.append(token.lemma_)  # Lemmatize

    return " ".join(cleaned_tokens)


def extract_dependency_relations(text):
    doc = nlp(text)
    dependencies = []
    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "ROOT"):
            dependencies.append(token.text.lower())
    return dependencies

df["Dependencies"] = df["Requirement"].apply(extract_dependency_relations)

G = nx.DiGraph()
for idx, row in df.iterrows():
    G.add_node(idx, text=row["Requirement"])

for idx, row in df.iterrows():
    for dep in row["Dependencies"]:
        for other_idx, other_row in df.iterrows():
            if dep in other_row["Requirement"].lower() and idx != other_idx:
                G.add_edge(other_idx, idx)

df["Dependency Score"] = [nx.pagerank(G).get(i, 0) for i in df.index]

# Step 4: Generate BERT Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = model.encode(df["Requirement"].tolist(), convert_to_tensor=True).cpu().numpy()

# Step 6: Incorporate User Ranking
df["Mean Rank"] = (df["Business Value"] + df["Urgency"]) / 2
scaler = MinMaxScaler()
df["Normalized Mean Rank"] = scaler.fit_transform(df[["Mean Rank"]])
df["Normalized Dependency Score"] = scaler.fit_transform(df[["Dependency Score"]])

# Step 7: Combine Features
combined_features = np.hstack((bert_embeddings, df[["Normalized Dependency Score", "Normalized Mean Rank"]].values))
print('combined')
# Step 8: Reduce Dimensionality using UMAP
umap = UMAP(n_components=2, random_state=42)
reduced_features = umap.fit_transform(combined_features)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyswarm import pso
import matplotlib.pyplot as plt


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyswarm import pso
import matplotlib.pyplot as plt

# Define PSO Objective Function
def pso_objective(k):
    k = int(np.round(k[0]))  # Convert float to integer

    if k < 2:
        return 1  # Avoid invalid clustering

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    labels = kmeans.fit_predict(reduced_features)

    return -silhouette_score(reduced_features, labels)  # Negative because PSO minimizes

# Set PSO Parameters
lb = [2]   # Min clusters
ub = [10]  # Max clusters

# Run PSO Once
best_k, _ = pso(
    pso_objective, lb, ub,
    swarmsize=20,  # Increased swarm size
    maxiter=50,   # Increased iterations
    omega=0.7,     # Balanced inertia
    phip=2.0,      # Cognitive weight
    phig=2.5       # Social weight
)

best_k = int(np.round(best_k[0]))  # Extract scalar before conversion
print(f" Best number of clusters found by PSO: {best_k}")

# ---- Elbow Method for Verification ----
inertia_values = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    kmeans.fit(reduced_features)
    inertia_values.append(kmeans.inertia_)

# Use the best_k found by PSO, NOT the last k in the loop
kmeans_final = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
df["Cluster"] = kmeans_final.fit_predict(reduced_features)  #  Correct cluster assignment

# ---- Elbow Method Plot ----
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method to Verify Best k")
plt.show()


# Step 12: Compute Cluster Quality Metrics
silhouette_avg = silhouette_score(reduced_features, df["Cluster"])
davies_bouldin = davies_bouldin_score(reduced_features, df["Cluster"])
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Step 13: Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df["Cluster"], palette="viridis")
plt.title("Optimized Clustering with PSO")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cluster")
plt.show()

# Step 14: Export Results
df.to_csv("z3.optimized_ga_clusters.csv", index=False)

def evaluate_classifier(classifier, clusters, features, labels, classifier_name):
    """
    Trains and evaluates a classifier on the given features and cluster labels.

    Parameters:
        classifier: The classification model (e.g., KNeighborsClassifier, XGBClassifier).
        clusters (array-like): Cluster assignments for each data point.
        features (array-like): Feature matrix for classification.
        labels (array-like): True labels for evaluation.
        classifier_name (str): Name of the classifier being evaluated.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics.
    """

    # Split Data for Training
    X_train, X_test, y_train, y_test = train_test_split(features, clusters,
                                                        test_size=0.4, stratify=clusters,
                                                        random_state=42)

    # Train Classifier
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    # Classification Report (includes precision, recall, F1)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']


    mse = mean_squared_error(y_test, y_pred)

    # Compute Top-3 Accuracy
    probs = classifier.predict_proba(X_test)
    top_3_predictions = np.argsort(probs, axis=1)[:, -3:]
    top_3_correct = sum([1 if y_test.iloc[i] in top_3_predictions[i] else 0 for i in range(len(y_test))])
    top_3_accuracy = top_3_correct / len(y_test)

    # Cross-Validation Score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, features, clusters, cv=skf)
    cross_val_mean = np.mean(cv_scores)

    # Compute Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(features, clusters)
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {classifier_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Create a DataFrame to store results
    results_df = pd.DataFrame([{
        'Classifier': classifier_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean Squared Error (MSE)': mse,
        'Top-3 Accuracy': top_3_accuracy,
        'Cross-Validation Score': cross_val_mean,
        'Davies-Bouldin Index': davies_bouldin

    }])

    return results_df
csv_filename = "Classification/3.classification_evaluation.csv"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
import os
# Define the list of classifiers to evaluate
classifiers = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', max_depth=4, learning_rate=0.05,
                             n_estimators=50, reg_lambda=0.1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, kernel='rbf', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
#    "Gaussian Naive Bayes": GaussianNB(),
#    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
#    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
#    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
#    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
#    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
#    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
#    "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
}

csv_filename = "Classification/3.smrat.csv"

# Loop through classifiers and evaluate each
for classifier_name, classifier in classifiers.items():
    results_df = evaluate_classifier(classifier, df["Cluster"], reduced_features, df["Cluster"], classifier_name)

    # Append results to CSV
    results_df.to_csv(csv_filename, index=False, mode='a', header=not os.path.exists(csv_filename))

    print(f"{classifier_name} evaluation completed and saved.")

