"""
Icci propagate module
"""
from scipy import stats
import numpy as np

from utils import ClassEncoder

class IcciPropagate:
    """
    Island classificator with confidence interval
    """
    def __init__(self, confidence: float = 0.999, exp: int = 2,
                 distribution: str = 'normal', unk_label: str = '<unk>'):
        """
        Icci initialization.
        """
        self.noclass_label = unk_label
        self.confidence = confidence
        self.distribution = distribution
        self.exp = exp
        self.statistics = {}
        self.encoder = ClassEncoder(unk_label=self.noclass_label)
        self.target = None

    def fit(self, feature_data: np.ndarray, labels: np.ndarray):
        """
        This distribution creates a dict using the statistics of train dataset.
        This statistics dict will be used to predict new rows.
        """
        feature_data = np.array(feature_data)
        labels = np.array(labels)
        self.encoder.fit(labels)
        self.target = self.encoder.transform(labels)
        for elabel in np.unique(self.target):
            self.statistics[elabel] = {}
            index = np.where(self.target == elabel, True, False)
            feature_label = feature_data[index]
            for key, column in enumerate(feature_label.T):
                if column.any():
                    mean, sigma = np.mean(column), np.std(column)
                    if self.distribution == 't-student':
                        conf_int = stats.t.interval(self.confidence, len(column), loc=mean,
                                                    scale=sigma)
                    else:
                        conf_int = stats.norm.interval(self.confidence, loc=mean, scale=sigma)

                    self.statistics[elabel][f'feature{key}'] = conf_int
                else:
                    self.statistics[elabel][f'feature{key}'] = (0, 0)



    def approximate_predict(self, features_data: np.ndarray):
        """
        Using the statistics dict, predict the label of new rows.
        """
        final_elabel = []
        for row in features_data:
            list_keys = list(self.statistics.keys())
            for index in range(row.shape[0]):
                for label in list_keys:
                    if not (row[index] >= self.statistics[label][f'feature{index}'][0]
                            and row[index] <= self.statistics[label][f'feature{index}'][1]):
                        list_keys.remove(label)

            if len(list_keys) == 1:
                final_elabel.append(list_keys[0])
            elif not list_keys:
                final_elabel.append(-1)
            elif len(list_keys) >= 1:
                result = -1
                metric_result = np.inf
                for label in list_keys:
                    stack = 0
                    for index in range(features_data.shape[1]):
                        limit1 = self.statistics[label][f'feature{index}'][0]
                        limit2 = self.statistics[label][f'feature{index}'][1]
                        stack += ((((limit2 + limit1) / 2) - row[index]) ** self.exp)
                    if metric_result > stack ** (1/self.exp):
                        result = label
                        metric_result = stack ** (1/self.exp)
                final_elabel.append(result)

        return self.encoder.inverse_transform(final_elabel)