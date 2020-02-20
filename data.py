# Add data generation functions here. This will include not just synthetic data, but also fetching real-world data from Yahoo! Finance etc.


import numpy as np
import pandas as pd
import random
from scipy.stats import norm, t

class gData():
        
    def genGaussian(nObs,sLength,size0,size1,mu0,sigma0,sigma1F):
        '''
        Produces a matrix of time series with a Gaussian distribution
        
        Parameters:
            nObs: 
            size0: number of vectors that are uncorrelated
            size1: number of vectors that are correlated
            
        
        
        '''
        #1) generate random uncorrelated data
        x=np.random.normal(mu0,sigma0,size=(nObs,size0)) # each row is a variable
        #2) create correlation between the variables
        cols=[random.randint(0,size0-1) for i in range(size1)]
        y=x[:,cols]+np.random.normal(0,sigma0*sigma1F,size=(nObs,len(cols)))
        x=np.append(x,y,axis=1)
        #3) add common random shock
        point=np.random.randint(sLength,nObs-1,size=2)
        x[np.ix_(point,[cols[0],size0])]=np.array([[-.5,-.5],[2,2]])
        #4) add specific random shock
        point=np.random.randint(sLength,nObs-1,size=2)
        x[point,cols[-1]]=np.array([-.5,2])
         
        return x,cols


class CorrelationGenerator:
    
    '''
    Generates synthetic correlation matrices.
    '''
    
    @classmethod
    def generateDiagnol(cls,n):
        '''
        Generates a correlation matrix with zero correlation
        
        Parameter:
            n: Size of correlation matrix
        Returns:
            res: correlation matrix
        '''
        
        return np.eye(n)
    
    @classmethod
    def generateRandom(cls,n):
        '''
        Generates a correlation matrix with random correlations
        
        Parameter:
            n: Size of correlation matrix
        Returns:
            res: correlation matrix
        '''
        A = np.random.normal(size=(n,n))
        res = A.T.dot(A)
        var = res[np.arange(n),np.arange(n)]
        return (res/np.sqrt(var.reshape(1,n)))/np.sqrt(var.reshape(n,1))

    @classmethod
    def isPosDef(cls, x):
        '''
        Checks if given correlation is positive-semi definite or not
        
        Parameters:
            x: 2-D ndarray
        Returns:
            bool: is x is positive-semi definite or not
        '''
        return np.all(np.linalg.eigvals(x) > 0)

    @classmethod
    def generateCorrelated(cls, n, rholim):
        '''
        Generate a correlation matrix where elements are correlated with correlations limited by rholims
        
        Parameters:
            n: size of covariance matrix
            rholim: tuple giving limits of correlated b/w elements
        Returns:
            res: 2-d ndarray, covariance matrix
        '''
        
        res = np.zeros(shape=(n,n))
        
        it = 0
        while not cls.isPosDef(res):
            print('Iteration: {}'.format(it))
            it += 1
            res = np.random.uniform(rholim[0],rholim[1],size=(n,n))
            res[np.arange(n),np.arange(n)] = 1
        
        return res

    @classmethod
    def generateCorrelatedGroups(cls, ns, rholims ):
        '''
        Generate a correlation matrix with certain groups having certain kind of correlations
        
        Parameters:
            ns: ndarray containing size of each correlated group
            rholims: list of tuples, each tuple gives limits of correlated b/w elements
        Returns:
            res: 2-d ndarray, covariance matrix
        '''
        
        n = ns.sum()
        ncum = ns.cumsum()
        res = np.zeros(shape=(n,n))
        
        print('Generate Group {}'.format(0))
        res[:ncum[0],:ncum[0]] = cls.generateCorrelated(ns[0],rholims[0])
        
        for i in range(1,len(ns)):
            print('Generate Group {}'.format(i))
            res[ ncum[i-1]:ncum[i], ncum[i-1]:ncum[i] ] = cls.generateCorrelated(ns[i],rholims[i])
        
        return res

    @classmethod
    def generateCorrelatedGroupsNeg(cls, ns, rholims, negrho, limit = 1000 ):
        '''
        Generate a correlation matrix with certain groups having certain kind of correlations
        
        Parameters:
            ns: ndarray containing size of each correlated group
            rholims: list of tuples, each tuple gives limits of correlated b/w elements
            negrho: negative correlation between elements of different groups
            limit: no.of iterations to try
        Returns:
            res: 2-d ndarray, covariance matrix
        '''
        
        n = ns.sum()
        ncum = ns.cumsum()
        res = np.zeros(shape=(n,n))
        
        it = 0
        while not cls.isPosDef(res):
            print('Iteration: {}'.format(it))
            it += 1
            print('Generate Group {}'.format(0))
            res[:,:] = negrho
            res[:ncum[0],:ncum[0]] = cls.generateCorrelated(ns[0],rholims[0])
            
            for i in range(1,len(ns)):
                print('Generate Group {}'.format(i))
                res[ ncum[i-1]:ncum[i], ncum[i-1]:ncum[i] ] = cls.generateCorrelated(ns[i],rholims[i])
            
            if it > limit:
                raise Exception('Unable to find covariance matrix with negative correlation = {}'.format(negrho))
        
        return res

class GaussianGenerator:
    '''
    Generates random variables with gaussian copula and gaussian marginals
    '''
    
    def __init__(self,var,mean,corr, *args, **kwargs):
        
        cov = None;
        
        if corr == 'Diagnol':
            cov = CorrelationGenerator.generateDiagnol(*args,**kwargs)
        elif corr == 'Random':
            cov = CorrelationGenerator.generateRandom(*args,**kwargs)
        elif corr == 'Correlated':
            cov = CorrelationGenerator.generateCorrelated(*args,**kwargs)
        elif corr == 'CorrelatedGroups':
            cov = CorrelationGenerator.generateCorrelatedGroups(*args,**kwargs)
        elif corr == 'CorrelatedGroupsNeg':
            cov = CorrelationGenerator.generateCorrelatedGroupsNeg(*args,**kwargs)
        else:
            raise Exception('Unknown Covariance Generator')
        
        self.var = var
        self.cov = cov*np.sqrt(var.reshape(-1,1))*np.sqrt(var.reshape(1,-1))
        self.mean = mean
    
    def generate(self,n=100):
        
        return np.random.multivariate_normal(self.mean,self.cov,size=n).T

class TGenerator(GaussianGenerator):
    '''
    Generates random variables with gaussian copula and standard_t marginals
    '''
    
    def __init__(self,df,var,mean,corr,*args,**kwargs):
        super(TGenerator,self).__init__(var,mean,corr,*args,**kwargs)
        
        self.df = df
        
    def generate(self,n=100):
        
        uni = norm.cdf(super(TGenerator,self).generate(n))
        
        return (t.ppf(uni,self.df.reshape(-1,1)) + self.mean.reshape(-1,1))*np.sqrt(self.var.reshape(-1,1))
        