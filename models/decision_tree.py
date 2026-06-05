import numpy as np

class Node:
    """A node in the custom decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, probability=None):
        self.feature = feature          # index of feature to split on
        self.threshold = threshold      # split threshold
        self.left = left                # left child
        self.right = right              # right child
        self.value = value              # prediction (class label) if leaf node
        self.probability = probability  # class probabilities if leaf node

    def is_leaf_node(self):
        return self.value is not None

    def to_dict(self):
        """Recursively serialize the node to a dictionary for JSON export."""
        if self.is_leaf_node():
            return {
                "type": "leaf",
                "value": int(self.value) if isinstance(self.value, (np.integer, int)) else self.value,
                "probability": {str(k): float(v) for k, v in self.probability.items()} if self.probability else {}
            }
        else:
            return {
                "type": "split",
                "feature": int(self.feature),
                "threshold": float(self.threshold) if isinstance(self.threshold, (np.floating, float)) else self.threshold,
                "left": self.left.to_dict(),
                "right": self.right.to_dict()
            }


class DecisionTreeClassifier:
    """
    A custom Decision Tree Classifier implemented from scratch using Numpy.
    Uses Gini Impurity for splitting criteria.
    """
    def __init__(self, max_depth=6, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        """Fit the decision tree on training data."""
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            return self._create_leaf_node(y)

        # Find the best split using Gini Impurity
        best_feat, best_thresh = self._best_split(X, y, n_samples, n_features)
        
        # If no split improves Gini impurity, create leaf node
        if best_feat is None:
            return self._create_leaf_node(y)

        # Split and create child nodes
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        
        # Check if split violates minimum leaf samples requirement
        if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
            return self._create_leaf_node(y)

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _create_leaf_node(self, y):
        """Create a leaf node containing the prediction value and probabilities."""
        vals, counts = np.unique(y, return_counts=True)
        most_common_idx = np.argmax(counts)
        value = vals[most_common_idx]
        
        # Compute probabilities
        total = len(y)
        probs = {}
        for c in self.classes_:
            c_count = np.sum(y == c)
            probs[c] = float(c_count) / total
            
        return Node(value=value, probability=probs)

    def _best_split(self, X, y, n_samples, n_features):
        """Find the best feature and threshold to split the dataset."""
        best_gini = 999.0
        best_feat, best_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # If feature is constant, skip it
            if len(thresholds) <= 1:
                continue
                
            # If there are too many thresholds, sample a subset to speed up training
            if len(thresholds) > 20:
                # Sample percentiles
                thresholds = np.percentile(X_column, np.linspace(5, 95, 20))

            for threshold in thresholds:
                gini = self._gini_impurity_split(X_column, y, threshold)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh

    def _gini_impurity_split(self, X_column, y, threshold):
        """Calculate Gini Impurity of a potential split."""
        left_idx = X_column <= threshold
        right_idx = X_column > threshold

        n_left = np.sum(left_idx)
        n_right = np.sum(right_idx)
        n_total = len(y)

        if n_left == 0 or n_right == 0:
            return 999.0

        # Calculate Gini for left split
        y_left = y[left_idx]
        gini_left = 1.0 - sum((np.sum(y_left == c) / n_left) ** 2 for c in self.classes_)

        # Calculate Gini for right split
        y_right = y[right_idx]
        gini_right = 1.0 - sum((np.sum(y_right == c) / n_right) ** 2 for c in self.classes_)

        # Weighted Gini
        weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return weighted_gini

    def predict(self, X):
        """Predict classes for input samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Recursively traverse the tree to get prediction for a sample."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def to_dict(self):
        """Export the decision tree to a dictionary structure."""
        if self.root is None:
            return {}
        return self.root.to_dict()
