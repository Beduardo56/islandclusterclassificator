from unittest import TestCase

import numpy as np

from utils import ClassEncoder


class IntegrationTestNoParams(TestCase):
    """
    Non-Parameterized tests simulate specific situations.
    """

    @staticmethod
    def test_class_encoder_numeric():
        """
        Simples test for the ClassEncoder class.
        """

        encoder = ClassEncoder()
        encoder.fit([1, 2, 2, 6])

        result_transf = encoder.transform([1, 1, 2, 6, encoder.unk_label])
        result_inv = encoder.inverse_transform([0, 0, 1, 2, -1])

        assert np.array_equal(encoder.classes_, np.array([1, 2, 6, encoder.unk_label]))
        assert encoder.classes_dict == {encoder.unk_label: -1, 1: 0, 2: 1, 6: 2}
        assert np.array_equal(result_transf, np.array([0, 0, 1, 2, -1]))
        assert np.array_equal(result_inv, np.array([1, 1, 2, 6, encoder.unk_label]))

    @staticmethod
    def test_class_encoder_strings():
        """
        Simples test for the ClassEncoder class.
        """

        unk_label = '<unk>'
        encoder = ClassEncoder(unk_label=unk_label)
        encoder.fit(["paris", "paris", "tokyo", "amsterdam", unk_label])

        result_transf = encoder.transform(["tokyo", "tokyo", "paris", unk_label])
        result_inv = encoder.inverse_transform([2, 2, 1, -1])

        assert np.array_equal(encoder.classes_,
                              np.array(['amsterdam', 'paris', 'tokyo', unk_label]))
        assert encoder.classes_dict == {unk_label: -1, 'amsterdam': 0, 'paris': 1, 'tokyo': 2}
        assert np.array_equal(result_transf, np.array([2, 2, 1, -1]))
        assert np.array_equal(result_inv, np.array(['tokyo', 'tokyo', 'paris', unk_label]))