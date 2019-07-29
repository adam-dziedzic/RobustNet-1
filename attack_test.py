import unittest
import numpy as np
import torch
from numpy.testing import assert_equal
from attack import class_ensemble


class TestUtils(unittest.TestCase):

    def test_class_ensemble(self):
        a = torch.tensor([[3, 2, 1], [1, 2, 1], [3, 2, 2]])
        a = a.numpy()
        print(a.shape, a)
        a = np.array(a)
        idx2 = class_ensemble(a)
        print(idx2)
        assert_equal(idx2.numpy(), [3, 2, 1])

    def test_class_ensemble2(self):
        m = np.array([[0, 2, 1], [1, 1, 0], [1, 2, 0]])
        idx2 = class_ensemble(m)
        print(idx2)
        assert_equal(idx2.numpy(), [1, 2, 0])


if __name__ == '__main__':
    unittest.main()