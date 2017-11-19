# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta,writeBeta,gradLogisticLoss,logisticLoss,lineSearch
from operator import add
from pyspark import SparkContext

def readDataRDD(input_file,spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)

         x is a dictionary of the form:

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The return value is an RDD containing tuples of the form
                 (SparseVector(x),y)

    """
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda (x,y):(SparseVector(x),y))




def getAllFeaturesRDD(dataRDD):
    """ Get all the features present in grouped dataset dataRDD.

	The input is:
            - dataRDD containing pairs of the form (SparseVector(x),y).

        The return value is an RDD containing the union of all unique features present in sparse vectors inside dataRDD.
    """
    return []

def totalLossRDD(dataRDD,beta,lam = 0.0):
    total_loss = dataRDD.map(lambda elem: logisticLoss(beta, elem[0], elem[1])).sum()
    return total_loss + lam * beta.dot(beta)

def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    grad_total_loss = dataRDD.map(lambda elem: gradLogisticLoss(beta, elem[0], elem[1])).\
        fold(SparseVector({}), add)
    return grad_total_loss + lam * beta

def test(dataRDD,beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.

        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0

                 TP = datapoints in (x,y) in P for which y=+1
                 FP = datapoints in (x,y) in P for which y=-1
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector β over data are defined as:

                 ACC(β,data) = ( #TP+#TN ) / (#P + #N)
                 PRE(β,data) = #TP / (#TP + #FP)
                 REC(β,data) = #TP/ (#TP + #FN)

        Inputs are:
             - data: an RDD containing pairs of the form (x,y),
                where x is a sparse vector and y is a binary value
             - beta: a sparse vector β

        The return values are
             - ACC, PRE, REC

    """
    scores = dataRDD.map(lambda elem: beta.dot(elem[0]))
    labels = dataRDD.map(lambda elem: elem[1])

    P = scores.map(lambda score: score > 0)
    N = scores.map(lambda score: score <= 0)

    TP = P.zip(labels).filter(lambda elem: elem[0] & (elem[1]==1)).count()
    FP = P.zip(labels).filter(lambda elem: elem[0] & (elem[1]==-1)).count()
    TN = N.zip(labels).filter(lambda elem: elem[0] & (elem[1]==-1)).count()
    FN = N.zip(labels).filter(lambda elem: elem[0] & (elem[1]==1)).count()

    ACC = (TP + TN) / (P.count() + N.count())
    PRE = TP / (TP + FP + 0.0001)
    REC = TP / (TP + FN + 0.0001)
    return ACC, PRE, REC

def train(dataRDD,beta_0,lam,max_iter,eps,test_data=None):
    """ Train a logistic classifier from deta.

	The function minimizes:

               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β||_2^2

        using gradient descent.

        Inputs are:
            - data: a RDD containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta_0: an initial sparse vector β_0
            - lam: the regularization parameter λ
            - max_iter: the maximum number of iterations
            - eps: the tolerance ε
            - test_data (optional): data over which model β is tested in each iteration w.r.t. accuracy, precision, and recall

        The return values are:
            - beta: the trained β, as a sparse vector
            - gradNorm: the norm ||∇L(β)||_2
            - k: the number of iterations

    """
    k = 0
    gradNorm = 2 * eps
    beta = beta_0
    start = time()
    while k < max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD, beta, lam)
        grad = gradTotalLossRDD(dataRDD, beta, lam)
        gradNormSq = grad.dot(grad)
        fun = lambda x: totalLossRDD(dataRDD, x, lam)
        gamma = lineSearch(fun, beta, grad, obj, gradNormSq)

        beta =  beta - gamma * grad
        if test_data == None:
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma
        else:
            acc,pre,rec = test(test_data,beta)
            print 'k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec
        k = k + 1

    return beta,gradNorm,k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')


    args = parser.parse_args()

    sc = SparkContext(appName='Parallel Logistic Regression')

    print 'Reading training data from',args.traindata
    traindata = readDataRDD(args.traindata, sc)
    print 'Read',traindata.count(),'data points with',len(getAllFeaturesRDD(traindata)),'features in total'

    if args.testdata is not None:
        print 'Reading test data from',args.testdata
        testdata = readDataRDD(args.testdata, sc)
        print 'Read',testdata.count(),'data points with',len(getAllFeaturesRDD(testdata)),'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    print 'Training on data from',args.traindata,'with λ =',args.lam,', ε =',args.eps,', max iter = ',args.max_iter
    beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps,test_data=testdata)
    print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
    print 'Saving trained β in',args.beta
    writeBeta(args.beta,beta)
