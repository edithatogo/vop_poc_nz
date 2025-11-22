import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterAnalysis:
    """
    Comprehensive cluster analysis for identifying intervention archetypes and cost-effectiveness patterns.
    """
    def __init__(self, all_results, models):
        self.all_results = all_results
        self.models = models
        self.cluster_results = {}

    def prepare_clustering_data(self, intervention_name, n_simulations=1000):
        """
        Prepare multi-dimensional data for clustering analysis.
        """
        print(f"Preparing clustering data for {intervention_name}...")

        # Get probabilistic results
        results = self.all_results[intervention_name]

        # Extract key features for clustering
        features = []

        for i in range(min(n_simulations, len(results['inc_cost']))):
            # Base cost-effectiveness metrics
            inc_cost = results['inc_cost'][i]
            inc_qalys = results['inc_qaly'][i]
            icer = inc_cost / inc_qalys if inc_qalys != 0 else float('inf')

            # Net monetary benefits at different WTP thresholds
            wtp_20k = (inc_qalys * 20000) - inc_cost
            wtp_50k = (inc_qalys * 50000) - inc_cost
            wtp_100k = (inc_qalys * 100000) - inc_cost

            # Cost components (initialize with defaults)
            primary_cost = 0
            productivity_cost = 0
            secondary_cost = 0

            if intervention_name == "HPV Vaccination":
                # HPV-specific features
                primary_cost = abs(np.random.normal(100, 20))  # Vaccine cost variation
                productivity_cost = abs(np.random.normal(2000, 500))  # Productivity impact
                secondary_cost = abs(np.random.normal(50000, 10000))  # Cancer treatment costs
            elif intervention_name == "Smoking Cessation":
                # Smoking cessation-specific features
                primary_cost = abs(np.random.normal(500, 100))  # Intervention cost
                productivity_cost = abs(np.random.normal(3000, 800))  # Productivity impact
                secondary_cost = abs(np.random.normal(10000, 2000))  # Downstream health costs
            else:  # Hepatitis C
                # Hepatitis C-specific features
                primary_cost = abs(np.random.normal(2000, 400))  # Treatment cost
                productivity_cost = abs(np.random.normal(4000, 1000))  # Productivity impact
                secondary_cost = abs(np.random.normal(5, 1))  # QALY improvement

            # Create feature vector
            feature_vector = [
                inc_cost,
                inc_qalys,
                icer if not np.isinf(icer) else 100000,  # Cap infinite ICERs
                wtp_20k,
                wtp_50k,
                wtp_100k,
                primary_cost,
                productivity_cost,
                secondary_cost
            ]

            features.append(feature_vector)

        return np.array(features)

    def perform_clustering(self, intervention_name, n_clusters_range=range(2, 6)):
        """
        Perform K-means clustering with optimal cluster number selection.
        """
        print(f"Performing clustering analysis for {intervention_name}...")

        # Prepare data
        features = self.prepare_clustering_data(intervention_name)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        features_pca = pca.fit_transform(features_scaled)

        # Find optimal number of clusters using silhouette score
        best_n_clusters = 2
        best_silhouette = -1
        best_kmeans = None

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)

            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(features_pca, cluster_labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n_clusters
                    best_kmeans = kmeans

        # Perform final clustering with optimal number
        if best_kmeans is None:
            # Fallback to 2 clusters if no good clustering found
            final_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            final_kmeans.fit(features_pca)
            best_silhouette = silhouette_score(features_pca, final_kmeans.predict(features_pca))
        else:
            final_kmeans = best_kmeans

        cluster_labels = final_kmeans.predict(features_pca)

        # Analyze cluster characteristics
        cluster_analysis = self.analyze_clusters(features, cluster_labels, intervention_name)

        # Store results
        self.cluster_results[intervention_name] = {
            'features': features,
            'features_scaled': features_scaled,
            'features_pca': features_pca,
            'cluster_labels': cluster_labels,
            'n_clusters': best_n_clusters,
            'cluster_centers': final_kmeans.cluster_centers_,
            'silhouette_score': best_silhouette,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'scaler': scaler,
            'pca': pca,
            'cluster_analysis': cluster_analysis
        }

        return self.cluster_results[intervention_name]

    def analyze_clusters(self, features, cluster_labels, intervention_name):
        """
        Analyze the characteristics of each cluster.
        """
        n_clusters = len(np.unique(cluster_labels))
        feature_names = [
            'Incremental Cost', 'Incremental QALYs', 'ICER',
            'NMB at $20k', 'NMB at $50k', 'NMB at $100k',
            'Primary Cost Component', 'Productivity Impact', 'Secondary Cost Component'
        ]

        cluster_analysis = {}

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_percentage = (cluster_size / len(cluster_labels)) * 100

            # Calculate cluster statistics
            cluster_means = np.mean(features[cluster_mask], axis=0)
            cluster_stds = np.std(features[cluster_mask], axis=0)

            # Identify key differentiators
            overall_means = np.mean(features, axis=0)
            overall_stds = np.std(features, axis=0)

            # Calculate standardized differences
            standardized_diffs = (cluster_means - overall_means) / overall_stds

            # Find most distinctive features
            distinctive_features = np.argsort(np.abs(standardized_diffs))[::-1][:3]

            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': cluster_size,
                'percentage': cluster_percentage,
                'means': cluster_means,
                'stds': cluster_stds,
                'distinctive_features': distinctive_features,
                'feature_names': [feature_names[i] for i in distinctive_features],
                'standardized_differences': standardized_diffs
            }

        return cluster_analysis
