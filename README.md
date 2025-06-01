# SAPC: Smart Agile Prioritization and Clustering

SAPC is an AI-driven framework designed to automate the prioritization and clustering of Agile software requirements. It integrates semantic embeddings, stakeholder input, and dependency analysis to deliver accurate, scalable sprint planning.

## Key Components
- **Text Preprocessing**: Removes noise and retains content words using NLP.
- **Semantic Embeddings**: Uses Sentence-BERT to capture deep requirement meaning.
- **Stakeholder Integration**: Combines business value and urgency for ranking.
- **Dependency Modeling**: Constructs PageRank-based graph from requirement relationships.
- **Clustering Optimization**: Uses Particle Swarm Optimization (PSO) to determine optimal clusters.
- **Evaluation**: Validated using Silhouette Score, Davies-Bouldin Index, F1-score, MSE, and Top-3 Accuracy.

