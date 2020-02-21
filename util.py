# Takes as input the model allocations, mean returns and mean variance of each portfolio, outputs 2 reports and a chart


import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew


class metrics():
    
    def __init__(self):
        self.rf = 0
        
    def adj_sharpe_ratio(self,data,weights):
        
        '''
        Paramters:
            data: Takes T x N np array of returns of N assets across T timeperiods
            weights: Takes T x N np array of weights of N assets across T timeperiods
            
        Returns:
            Adjusted Sharpe ratio for portfolio returns for the time period
            
        '''
        
        assert weights.shape == data.shape, "Check input vector shapes"
        
        returns = np.sum(weights*data,axis = 1)
        
        assert returns.shape[0] == weights.shape[0]
        
        SR = (np.mean(returns) - self.rf)/np.std(returns)
                
        ASR = SR*(1 + (skew(returns)/6)*SR - ((kurtosis(returns) - 3)/24)*SR**2)
        
        return ASR
    
    def cert_equivalent_ret(self,data,weights,gamma = 1):
        
        '''
        Paramters:
            data: Takes T x N np array of returns of N assets across T timeperiods
            weights: Takes T x N np array of weights of N assets across T timeperiods
            
        Returns:
            The certainty-equivalent return (CEQ): The risk-free rate of 
            return that the investor is willing to accept instead of 
            undertaking the risky portfolio strategy.
            
        '''
        
        assert weights.shape == data.shape, "Check input vector shapes"
        
        returns = np.sum(weights*data,axis = 1)
        
        assert returns.shape[0] == weights.shape[0]
        
        CEQ = (np.mean(returns) - self.rf) - (gamma*np.std(returns)**2)/2
        
        return CEQ
    
    
    def avg_turnover(self,weights):
        
        '''
        
        Paramters:
            weights:Takes T x N np array of weights of N assets across T timeperiods
            
            *** Include t = 0 weights to if rebalancing starts at t = 1 ***
            
        Returns:
            Average Turnover per rebalancing
        
            *** Assumes that the portfolio is rebalanced daily (for all timeperiods) ***
            *** If a non-daily measure is needed, then input corresponding weights vector ***
            
        '''
    
        return np.mean(np.sum(np.absolute(weights[1:,:] - weights[:-1,:]),axis = 1))
        
    
    def sum_sq_port_wts(self,weights):
        
        '''
        Paramters:
            weights:Takes T x N np array of weights of N assets across T timeperiods
            
        Returns:
            The sum of squared portfolio weights (SSPW): Exhibits the underlying 
            level of diversification in a portfolio
        
        '''
        
        return np.mean(np.sum(weights**2,axis = 1))
  

if __name__ == '__main__': 
    
    # Sample tests

    wts = np.array([[0.1,0.2,0.3,0.4],[0.2,0.1,0.3,0.4],[0.4,0.1,0.2,0.3]])
    data = np.array([[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2],[0.2,0.3,0.4,0.2]])

    
    print(metrics().adj_sharpe_ratio(data,wts))

    print(metrics().cert_equivalent_ret(data,wts))

    print(metrics().avg_turnover(wts))

    print(metrics().sum_sq_port_wts(wts))
        
        
        
        
        
        
        