"""
ML Utils library.
"""

from typing import Iterable

import numpy as np

class ClassEncoder:
    """
    Classes encoder class.
    """

    def __init__(self, unk_label: str = None):
        self.unk_label = unk_label
        self.classes_dict = None
        self.classes_ = None

    def fit(self, target_names: Iterable):
        """
        Encodes the target_names labels.
        """

        target_names = np.array(target_names)

        # Setting the classes_ array:
        target_labels = []
        classes_dict = {self.unk_label: -1}
        target_count = 0
        for label in np.unique(target_names):
            if label != self.unk_label:
                target_labels.append(label)
                classes_dict[label] = target_count
                target_count += 1
        self.classes_ = np.array(target_labels + [self.unk_label])
        self.classes_dict = classes_dict

    def transform(self, target_names: Iterable) -> np.ndarray:
        """
        Encodes the target_names labels into indexes.
        """

        targets = np.array([self.classes_dict[label] for label in target_names])

        return targets

    def inverse_transform(self, y_pred: Iterable) -> np.ndarray:
        """
        Encodes the indexes into target_names labels.
        """

        y_pred = np.array(y_pred, dtype='int64')
        target_labels = self.classes_[y_pred]

        return target_labels
