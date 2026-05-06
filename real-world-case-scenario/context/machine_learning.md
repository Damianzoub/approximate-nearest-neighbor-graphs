# Machine Learning

Machine learning (ML) is a subfield of artificial intelligence that enables systems to learn from data and improve their performance on tasks without being explicitly programmed. Instead of hand-crafting rules, ML algorithms identify patterns in data and build statistical models that generalise to new, unseen examples.

## Supervised Learning

In supervised learning, the algorithm learns from labelled training data — input-output pairs — to predict outputs for new inputs. Classification assigns inputs to discrete categories (spam vs. not spam, disease vs. healthy). Regression predicts continuous values (house prices, temperature, stock returns).

Key algorithms include linear and logistic regression (simple but interpretable), decision trees, random forests, and gradient boosting machines (powerful ensembles). Support Vector Machines (SVMs) find the optimal hyperplane separating classes. Neural networks, particularly deep networks, achieve state-of-the-art performance on high-dimensional data like images, audio, and text.

Overfitting — when a model memorises training data rather than learning general patterns — is a central challenge. Regularisation techniques (L1/L2 penalties, dropout, data augmentation), cross-validation, and early stopping help produce models that generalise well.

## Unsupervised Learning

Unsupervised learning discovers structure in unlabelled data. Clustering algorithms (k-means, DBSCAN, hierarchical clustering) group similar data points. Dimensionality reduction techniques (PCA, t-SNE, UMAP) project high-dimensional data into lower-dimensional spaces for visualisation and downstream modelling.

Generative models learn the underlying data distribution. Variational Autoencoders (VAEs) learn compact latent representations and generate new samples. Generative Adversarial Networks (GANs), consisting of a generator and discriminator in adversarial training, produce remarkably realistic images, audio, and video.

## Reinforcement Learning

Reinforcement learning (RL) trains agents to maximise cumulative reward through trial-and-error interaction with an environment. The agent takes actions, receives rewards or penalties, and updates its policy to favour actions that lead to higher rewards. Q-learning and policy gradient methods are foundational RL algorithms.

Deep RL, combining RL with deep neural networks, achieved landmark results: AlphaGo defeated the world Go champion; AlphaStar reached Grandmaster level in StarCraft II; OpenAI Five defeated professional Dota 2 teams. RL is applied in robotics, autonomous driving, resource allocation, and drug discovery.

## Approximate Nearest Neighbour Search in ML

Many ML systems rely on finding similar items quickly in high-dimensional vector spaces — retrieval-augmented generation, recommendation systems, image search, and embedding-based classification all require efficient nearest neighbour search. Exact nearest neighbour search scales poorly as dataset size and dimensionality grow.

Approximate Nearest Neighbour (ANN) algorithms trade a small loss in accuracy for dramatic gains in speed. Hierarchical Navigable Small World (HNSW) graphs achieve near-linear query time and are the dominant algorithm in production vector databases. Algorithms like DARTH add learned early termination to HNSW using LightGBM predictors, while PiP (Point in Polytope) uses geometric stability checks to prune the search. Ada-ef dynamically selects the search expansion factor based on query distribution.

## Model Evaluation and Deployment

Rigorous model evaluation requires splitting data into training, validation, and test sets, using appropriate metrics (accuracy, F1, AUC-ROC, RMSE), and validating on realistic distributions. Model interpretability tools — SHAP values, LIME, attention maps — help understand model behaviour and detect biases.

MLOps practices — continuous integration, automated testing, model versioning, monitoring, and retraining pipelines — ensure that ML systems remain accurate and reliable in production as data distributions shift over time.
