# Add data generation functions here. This will include not just synthetic data, but also fetching real-world data from Yahoo! Finance etc.


import numpy as np
import pandas as pd
import random

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
    