from __future__ import annotations
from dataclasses import dataclass
from pprint import pprint
import collections
from typing import Tuple
import numpy as np


@dataclass
class Node:
    
    split_feature_idx: int = None
    split_threshold: float = None
    l_child: Node = None
    r_child: Node = None
    majority_class: int = None

    @property
    def is_leaf(self):
        return True if self.majority_class is not None else False

@dataclass
class ClassificationTree:

    """Builds a binary classification tree using Gini-index as impurity criterion for best splits."""
    
    nmin: int = 1   #minimum samples in a node to consider split, if < then node becomes leaf
    minleaf: int = 1   #minimum required leaf size after splitting of a node
    nfeats: int = None   #number of features to consider in a split (set to None, i.e. all features for regular classification trees)
    
    def __post_init__(self):
        assert self.nmin > 0
        assert self.minleaf > 0
        self.tree = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        assert isinstance(X, np.ndarray), "X is not an n-dimensional numpy array. If X is a list of lists, convert it to an n-dimensional numpy array."
        assert isinstance(y, np.ndarray), "y is not an n-dimensional numpy array. If y is a list, convert it to a 1-dimensional numpy array."
        assert np.ndim(y) == 1, "Dimension of y is not 1."
        assert len(X) == len(y), "The number of rows in X is not equal to the size of y."
        
        # nfeats to use each split. Else clause ensures that nfeats used cannot exceed the total nfeats of the data (in case a user accidentally specifies more)
        if self.nfeats is None:
            self.nfeats = X.shape[1]
        else:
            assert self.nfeats > 0, "nfeats should be a positive integer."
            self.nfeats = min(self.nfeats, X.shape[1])
        
        self.Xcol_dim = X.shape[1] # Store the X column dimension for an error check in predict func
        
        self.tree_grow(X, y)

    def tree_grow(self, X, y) -> None:
        
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
        parent_node = self.create_internal_node(best_feature_idx, best_threshold, l_child, r_child)  #node object
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
        self.tree.append(leaf_node)
        return leaf_node 

    def create_internal_node(self, split_feature_idx, best_thresh, l_child, r_child) -> Node:
        internal_node = Node(split_feature_idx, best_thresh, l_child, r_child)
        self.tree.append(internal_node)
        return internal_node

    def majority_class(self, y: np.ndarray) -> int:
        if len(np.unique(y)) > 1:
            return max(collections.Counter(y), key=lambda k: y[k]) 
        else:
            return max(collections.Counter(y))
         
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Traverse the tree starting at root until arrived at a leaf node. When arrived at a leaf node, 
        return the majority class prediction."""

        assert X.shape[1] == self.Xcol_dim, "Input X column dim does not correspond with the X column dim used for training."
        
        predicted_y = []
        for obs in X:
            
            current_node = self.tree[-1] # start the root node
            while not current_node.is_leaf:
                
                if obs[current_node.split_feature_idx] <= current_node.split_threshold:
                    current_node = current_node.l_child
                else:
                    current_node = current_node.r_child 

            else:
                predicted_y.append(current_node.majority_class)
            
        return np.asarray(predicted_y)


class RandomForestClassifier:
    
    pass


if __name__ == '__main__':  
    data = np.genfromtxt("pima-indians-diabetes.csv", delimiter=',')
    X, y = data[:, 0:8], data[:, 8]
    y = y.astype('int64')
    # print(X[0:5, :])
    # print(y[0:5])

    clf = ClassificationTree()
    clf.fit(X, y)
    # pprint(clf.tree[-1])

    X_test = np.array([[1,103,30,38,83,43.3,0.183,33]])
    print(clf.predict(X_test))
