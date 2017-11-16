import unittest

from pyspark import SparkContext

import SparseVector
import LogisticRegression
import ParallelLogisticRegression



class SparseVectorTestCase(unittest.TestCase):
    def test_p_norm(self):
        sparse = SparseVector.SparseVector({'a': 2, 'b': 2})

class LogsticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        self.x = SparseVector.SparseVector(dict(a=1,b=1))
        self.y = 1
        self.beta = SparseVector.SparseVector(dict(a=3,b=3))
    def test_logisticLoss(self):
        kek = LogisticRegression.logisticLoss(self.beta, self.x, self.y)
    def test_gradLogisticLoss(self):
        grad = LogisticRegression.gradLogisticLoss(self.beta, self.x, self.y)
    def test_gradTotalLoss(self):
        grad = LogisticRegression.gradTotalLoss([(self.x, self.y)], self.beta)
    def test_test(self):
        data = [
        (SparseVector.SparseVector({'a': -1, 'b': -1}), -1),
        (SparseVector.SparseVector({'a': -1, 'b': -1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1)
        ]
        beta = SparseVector.SparseVector({'a': 2, 'b': 2})
        scores = LogisticRegression.test(data, beta)


class ParallelLogisticRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(ParallelLogisticRegressionTestCase, cls).setUpClass()
        cls.sc = SparkContext(appName="ParallelLogisticRegressionTestCase")

    @classmethod
    def tearDownClass(cls):
        super(ParallelLogisticRegressionTestCase, cls).tearDownClass()
        cls.sc.stop()

    def test_gradTotalLoss(self):
        rdd = self.sc.parallelize([
        (SparseVector.SparseVector({'a': -1, 'b': -1}), -1),
        (SparseVector.SparseVector({'a': -1, 'b': -1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1),
        (SparseVector.SparseVector({'a':1, 'b': 1}), 1)
        ])
        beta = SparseVector.SparseVector({'a':2, 'b':3})
        kek = ParallelLogisticRegression.gradTotalLossRDD(rdd, beta)
        print kek
