from sklearn.metrics import silhouette_score
import numpy as np

class UsagePatternClustering:
    # ...existing code...
    def train(self, X, labels):
        # Sample if dataset is large
        if X.shape[0] > 10000:
            sample_idx = np.random.choice(X.shape[0], 10000, replace=False)
            score = silhouette_score(X[sample_idx], labels[sample_idx])
        else:
            score = silhouette_score(X, labels)
        # ...existing code...
