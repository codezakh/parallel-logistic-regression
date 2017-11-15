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
    pass   

def totalLossRDD(dataRDD,beta,lam = 0.0):
    pass

def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    pass  



def test(dataRDD,beta):
    pass

def train(dataRDD,beta_0,lam,max_iter,eps,test_data=None):
    pass

if __name__ == "__main__":
    pass
