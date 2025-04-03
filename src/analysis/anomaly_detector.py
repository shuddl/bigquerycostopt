# ...existing code...
# Replace Python loop for z-score calculation
z_scores = (df - df.mean()) / df.std()
# ...existing code...

class AnomalyDetector:
    def __init__(self):
        # ...existing code...
        self._feedback_buffer = []
        self.update_threshold = 100  # Process feedback in batches of 100

    def update_with_feedback(self, new_feedback):
        self._feedback_buffer.append(new_feedback)
        if len(self._feedback_buffer) >= self.update_threshold:
            self._process_feedback_batch(self._feedback_buffer)
            self._feedback_buffer = []

    def _process_feedback_batch(self, feedback_batch):
        # Process the batch of feedback
        # ...existing code...
