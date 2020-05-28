"""
Tests for the module icciscan.py module.
"""

from unittest import TestCase

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.iccipropag import IcciPropagate


class UnitTestNoParams(TestCase):
    """
    Non-Parameterized tests simulate specific situations.
    """

    @staticmethod
    def test_icciscan_fit1():
        """
        Test for the icciscan.fit() method.
        """

        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18], [0, 0]]
        target_names = np.array(['a', 'b', 'c', 'd', '<unk>'])

        icci = IcciPropagate(unk_label='<unk>')
        icci.fit(data, target_names)

        assert isinstance(icci.statistics, dict)
        assert len(icci.statistics) == 5
        assert np.array_equal(icci.target, [0, 1, 2, 3, -1])
        assert np.array_equal(icci.encoder.classes_, ['a', 'b', 'c', 'd', '<unk>'])

    @staticmethod
    def test_icciscan_fit_iris():
        """
        Test for the icciscan.fit() method on the iris dataset.
        """

        iris = load_iris()
        target_names = iris['target_names'][iris['target']]

        icci = IcciPropagate()
        icci.fit(iris['data'], target_names)

        assert isinstance(icci.statistics, dict)
        assert len(icci.statistics) == 3
        assert np.array_equal(icci.target, iris['target'])
        assert np.array_equal(icci.encoder.classes_, ['setosa', 'versicolor', 'virginica', '<unk>'])


class IntegrationTestNoParams(TestCase):
    """
    Class for integration tests
    """

    @staticmethod
    def test_icciscan_approximate_predict():
        """
        Simples test for the icciscan.approximate_predict() method.
        """

        data = np.array([[1, 9], [1, 11], [2, 10], [0, 10],
                         [10, 1], [10, 3], [9, 2], [11, 2],
                         [5, 5], [0, 0], [10, 10]])
        target_names = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b',
                                 '<unk>', '<unk>', '<unk>'])

        icci = IcciPropagate(unk_label='<unk>')
        icci.fit(data, target_names)

        test_points = np.array([[1, 10], [10, 2], [5, 5]])
        test_labels = icci.approximate_predict(test_points)

        assert isinstance(icci.statistics, dict)
        assert isinstance(icci.encoder.classes_, np.ndarray)
        assert np.array_equal(icci.encoder.classes_, ['a', 'b', '<unk>'])
        assert isinstance(test_labels, np.ndarray)
        assert np.array_equal(test_labels, ['a', 'b', '<unk>'])


    @staticmethod
    def test_icciscan_approximate_predict_iris():
        """
        Test for the icciscan.approximate_predict() method on the iris dataset.
        """

        iris = load_iris()
        x_array = iris['data']
        y_array = iris['target_names'][iris['target']]

        x_train, x_test, y_train, y_test = train_test_split(x_array, y_array,
                                                            test_size=0.6, random_state=42)

        icci = IcciPropagate(unk_label='<unk>')
        icci.fit(x_train, y_train)

        test_labels = icci.approximate_predict(x_test)
        assert isinstance(test_labels, np.ndarray)
        assert accuracy_score(y_test, test_labels) > 0.75
        assert f1_score(y_test, test_labels, average='weighted') > 0.85
        assert precision_score(y_test, test_labels, average='weighted') > 0.94
        assert recall_score(y_test, test_labels, average='weighted') > 0.75