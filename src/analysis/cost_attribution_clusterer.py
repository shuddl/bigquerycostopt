class CostAttributionClusterer:
    def __init__(self):
        # ...existing code...
        self._cached_scaled_features = None

    def preprocess(self, data):
        if self._cached_scaled_features is None:
            self._cached_scaled_features = self.scaler.fit_transform(data)
        return self._cached_scaled_features
# ...existing code...
