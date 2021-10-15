"""
For building classification trees and Random forest classifiers.
Call `fit` method on the classifier object with 2 numpy arrays 
of your X and y training data as input parameters. Call the `predict` 
method on the object using as input parameter the X test data in order
to make predictions on unseen data.

Made by:
    Etjen Ahmic (5617030)
    Kan Li (2216612)
    Ziyuan Wang ()
"""

from __future__ import annotations
from dataclasses import dataclass
import collections
from typing import Tuple, Optional
import numpy as np
import time
import math
import multiprocessing as mp


@dataclass
class Node:
    
    majority_class: int = None
    split_feature_idx: int = None
    split_threshold: float = None
    l_child: Node = None
    r_child: Node = None

    @property
    def is_leaf(self):
        return True if self.l_child is None and self.r_child is None else False

class ClassificationTree:

    """Builds a binary classification tree using Gini-index as impurity criterion for best splits."""
    
    def __init__(self, nmin: int = 2, minleaf: int = 1) -> None:
        self.nmin = nmin
        self.minleaf = minleaf
        assert self.nmin > 1
        assert self.minleaf > 0
        
    def fit(self, X: np.ndarray, y: np.ndarray, nfeats: Optional[int] = None) -> None:

        """Call this function to fit the classifier."""

        assert isinstance(X, np.ndarray), "X is not an n-dimensional numpy array. If X is a list of lists, convert it to an n-dimensional numpy array."
        assert isinstance(y, np.ndarray), "y is not an n-dimensional numpy array. If y is a list, convert it to a 1-dimensional numpy array."
        assert np.ndim(y) == 1, "Dimension of y is not 1."
        assert len(X) == len(y), "The number of rows in X is not equal to the size of y."
        
        # nfeats to use each split. Else clause ensures that nfeats used cannot exceed the total nfeats of the data (in case a user accidentally specifies more)
        self.nfeats = nfeats
        if self.nfeats is None:
            self.nfeats = X.shape[1]
        else:
            assert self.nfeats > 0, "nfeats should be a positive integer."
            self.nfeats = round(min(self.nfeats, X.shape[1]))
        
        self.Xcol_dim = X.shape[1] # Store the X column dimension for an assertion check in predict func
        
        self.tree = self.tree_grow(X, y) # Start growing the tree. The whole tree is stored in self.tree 

    def tree_grow(self, X: np.ndarray, y: np.ndarray) -> Node:
        
        """Recursively grow the decision tree. If a stopping criterium is met, the recursive function returns a 
        leaf Node, else an internal Node object (parent node). The Nodes are stored in the 'tree' instance variable. """
        
        n_obs, n_unique_classes = X.shape[0], len(np.unique(y))
        
        # Check if stopping criteria are met for the current node in the call stack. If not instantiate a leaf Node object
        if n_obs < self.nmin or n_unique_classes == 1:
            leaf_node = self.create_leaf_node(y)
            return leaf_node
        
        # Exhaustive search for finding the best split
        best_feature_idx, best_threshold = self.find_best_split(X, y) 
        
        # If these are None, then no single split was found that satisfied the minleaf constraint
        if (best_feature_idx, best_threshold) == (None, None):
            leaf_node = self.create_leaf_node(y)
            return leaf_node

        # Recursively grow the child nodes after a split (left child first)
        l_child_idx = np.argwhere(X[:, best_feature_idx] <= best_threshold).flatten()  #indices
        r_child_idx = np.argwhere(X[:, best_feature_idx] > best_threshold).flatten()  #indices
        l_child = self.tree_grow(X[l_child_idx, :], y[l_child_idx])  # node object
        r_child = self.tree_grow(X[r_child_idx, :], y[r_child_idx])  # node object
        parent_node = self.create_internal_node(y, best_feature_idx, best_threshold, l_child, r_child)  #node object
        return parent_node

    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        
        """Exhaustive search for finding the best feature and threshold to split on."""
        
        # Select only a sample of features to consider for the best split in the node (applies only to Random forest)
        X_indices = np.random.choice(X.shape[1], self.nfeats, replace=False) 
        
        best_impurity_reduction = -99999 # Negative placeholder value (if set to 0 may throw error down the line if there is a split with 0 impurity reduction)
        best_feature_idx, best_threshold = None, None
        
        # Loop over all column indices
        for idx in X_indices:
            
            # Subset the column and sort the values. Compute the midway thresholds
            X_col = X[:, idx]
            sorted_values = np.sort(np.unique(X_col))
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
            
            # Loop over all thresholds and find the best (weighted) gini impurity reduction
            for threshold in thresholds:
                
                impurity_reduction = self.gini_impurity_reduction(X_col, y, threshold)
                
                if impurity_reduction is not None:
                    if impurity_reduction > best_impurity_reduction:
                        best_impurity_reduction = impurity_reduction
                        best_feature_idx = idx
                        best_threshold = threshold

        return best_feature_idx, best_threshold
    
    @staticmethod
    def gini_index(y: np.ndarray) -> float:
        return 1 - sum((np.bincount(y)/len(y))**2) 

    def gini_impurity_reduction(self, X_col: np.ndarray, y: np.ndarray, threshold: float) -> float:
        
        """Computes the gini impurity reduction for a split."""
         
        # Get the observations (indices) belonging to the left and right child nodes for the tested split
        l_child_idx = np.argwhere(X_col <= threshold).flatten()
        r_child_idx = np.argwhere(X_col > threshold).flatten()
        
        # Check if resulting child nodes satisfy minleaf constraint
        if len(l_child_idx) < self.minleaf or len(r_child_idx) < self.minleaf:
            return None
        else:
            gini_parent = ClassificationTree.gini_index(y)
        
            # Compute gini impurity for the child nodes
            l_child_gini = ClassificationTree.gini_index(y[l_child_idx])
            r_child_gini = ClassificationTree.gini_index(y[r_child_idx])
            
            # Compute weighted avg gini impurity for the split
            weighted_avg_gini = len(y[l_child_idx])/len(y) * l_child_gini + len(y[r_child_idx])/len(y) * r_child_gini

            return gini_parent - weighted_avg_gini
            
    def create_leaf_node(self, y: np.ndarray) -> Node:
        majority_class = self.majority_class(y)
        leaf_node = Node(majority_class=majority_class)
        return leaf_node 

    def create_internal_node(self, y, split_feature_idx, best_thresh, l_child, r_child) -> Node:
        majority_class = self.majority_class(y)
        internal_node = Node(majority_class, split_feature_idx, best_thresh, l_child, r_child)
        return internal_node

    def majority_class(self, y: np.ndarray) -> int:
        majority_class = collections.Counter(y).most_common(1)[0][0]
        return majority_class
         
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        """Traverse the tree starting at root until arrived at a leaf node. When arrived at a leaf node, 
        return the majority class prediction."""

        assert X.shape[1] == self.Xcol_dim, "Input X column dim does not correspond with the X column dim used for training."
        
        predicted_y = []
        for obs in X:
            
            current_node = self.tree # Starts at the root node
            while not current_node.is_leaf:
                
                if obs[current_node.split_feature_idx] <= current_node.split_threshold:
                    current_node = current_node.l_child
                else:
                    current_node = current_node.r_child 

            else:
                predicted_y.append(current_node.majority_class)
            
        return np.array(predicted_y)


class RandomForestClassifier:
    
    """Class for building a Random Forest classifier."""

    def __init__(self, ntrees: int = 100, nmin: int = 2, minleaf: int = 1) -> None:
        self.ntrees = ntrees #ntrees = m (no. bootstrap samples)
        self.nmin = nmin
        self.minleaf = minleaf
        assert self.ntrees > 1, "ntrees needs to be at least 2."
        assert self.nmin > 1
        assert self.minleaf > 0

    def fit(self, X: np.ndarray, y: np.ndarray, nfeats: Optional[int] = None) -> None:

        """Fit the Random forest classifier. Creates a multiprocessing pool to speed up tree building."""

        self.nfeats = nfeats
        if self.nfeats is None:
            # Use sqrt of total predictor variables (Random Forest)
            self.nfeats = round(math.sqrt(X.shape[1]))
        else:
            assert self.nfeats > 0
            self.nfeats = round(min(self.nfeats, X.shape[1]))

        self.Xcol_dim = X.shape[1] # Store the X column dimension for an assertion check in predict func

        # Create ntrees number of bootstrap samples
        # bootstrap_samples is a list of tuples with each tuple a bootstrap sample (X, y) used to train a tree
        bootstrap_samples = []
        for _ in range(self.ntrees):
            bootstrap_samples.append(self.build_bootstrap_sample(X, y))
        
        # Start multiprocessing pool
        p = mp.Pool(processes=mp.cpu_count())
        self.clfs = p.map(self.tree_grow, bootstrap_samples)
        p.close()
        p.join()

    def tree_grow(self, data: Tuple[np.ndarray, np.ndarray]) -> ClassificationTree:
        
        """Grow the trees using bootstrapped samples."""

        X, y = data
        clf = ClassificationTree(nmin=self.nmin, minleaf=self.minleaf)
        clf.fit(X, y, nfeats=self.nfeats)
        return clf

    def build_bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        """Builds a bootstrap sample of the data."""

        indices = np.random.choice(len(X), size=len(X), replace=True)
        bootstrap_sample = X[indices], y[indices]
        return bootstrap_sample

    def predict(self, X: np.ndarray) -> int:

        """Traverse all constructed trees and return majority vote class prediction."""

        assert X.shape[1] == self.Xcol_dim, "Input X column dim does not correspond with the X column dim used for training."
        
        predictions_y = np.empty((len(X), self.ntrees), dtype=int)
        j = 0
        for clf in self.clfs:
            predictions_y[:, j] = clf.predict(X)
            j += 1
        
        majority_vote = []
        for row in predictions_y:
            most_predicted_class = collections.Counter(row).most_common(1)[0][0]
            majority_vote.append(most_predicted_class)

        return np.array(majority_vote)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # Testing only (pima indians diabetes dataset), see the Jupyter notebook for models trained on bug data set.
    data = np.genfromtxt("pima-indians-diabetes.csv", delimiter=',')
    X, y = data[:, 0:8], data[:, 8]
    y = y.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    t1 = time.time()

    rf_clf = RandomForestClassifier(nmin=20, minleaf=5)
    rf_clf.fit(X_train, y_train)

    t2 = time.time()
    print(f'Time: {t2-t1:.2f}s')

    y_pred = rf_clf.predict(X_test)

    cf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cf_matrix.ravel()
    print(cf_matrix)

    accuracy = (tn + tp)/(tn + tp + fp + fn)
    print(f'Accuracy: {accuracy:.2f}')

#    print(y_pred)

#    clf = ClassificationTree()
#    clf.fit(X_train, y_train)
#
#    y_pred = clf.predict(X_test)
