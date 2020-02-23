# Takes as input the model allocations, mean returns and mean variance of each portfolio, outputs 2 reports and a chart


import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, binned_statistic
from tabulate import tabulate

import matplotlib.pyplot as plt


class metrics():
    
    def __init__(self):
        self.rf = 0
     
        
    def sharpe_ratio(self,data,weights):
        
        '''
        Paramters:
            data: Takes T x N np array of returns of N assets across T timeperiods
            weights: Takes T x N np array of weights of N assets across T timeperiods
            
        Returns:
            Sharpe ratio for portfolio returns for the time period
        '''
       
        
        assert weights.shape == data.shape, "Check input vector shapes"
        
        returns = np.sum(weights*data,axis = 1)
        
        assert returns.shape[0] == weights.shape[0]
        
        SR = (np.mean(returns) - self.rf)/np.std(returns)
        
        return SR
        
        
        
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
  

class plotUtil():
    
    @classmethod
    def plot_wts_timeseries(self,res,names):
        
        '''
        Paramters: 
            res: Takes a list of tuples for each iteration. Each tuple rebalancing weights,
        and daily returns.
                - Weights: M x N x T np array - M is models, N is assets , T timeperiods
                - Daily rets: M x T np array - M is models, T is timeperiods
        
            names: List of strings with model names (in order)
        
        
        Plots a timeseries of weights across 'all' assets for all rebalancing periods
        '''
        
        assert len(res) > 0 , "Run experiment first!!"
        
        wts, dailyrets = res[0]
        
        wts_sum = np.zeros(wts.shape)
        dailyrets_sum = np.zeros(dailyrets.shape)
        
        for i in range(len(res)):
            wts, dailyrets = res[i]
            wts_sum += wts
            dailyrets_sum += dailyrets
            
        #Caclulate average across all simulations
        avg_wts = wts_sum/len(res)
        
        #For all models, plot graphs
        for m in range((wts.shape[0])):
            plt.figure()
            df = pd.DataFrame(avg_wts[m].T)
            df.plot.line()
            
            model_name = 'Model - ' + names[m]
            plt.title(model_name)
            plt.xlabel('Rebalancing period')
            plt.ylabel('Asset weights')
            plt.ylim(0.0,1.0)
            plt.show()
    
    @classmethod
    def gen_summary_statistics(self,res,names,return_df = False):
    
        '''
        Paramters: Takes a list of tuples for each iteration. Each tuple rebalancing weights,
        and daily returns.
        Weights: M x N x T np array - M is models, N is assets , T timeperiods
        Daily rets: M x T np array - M is models, T is timeperiods
        
            
        Creates a table of summary stats
        
        '''
        
        assert len(res) > 0 , "Run experiment first!!"
            
        wts, dailyrets = res[0]
            
        M,N,T = wts.shape
        n_sims = len(res)
                
        stats = {model : [] for model in names }
        
        for m in range(M):
            iter_turnover = []
            iter_sspw = []
            for i in range(n_sims):
                wts, dailyrets = res[i]
                
                iter_turnover.append(metrics().avg_turnover(wts[m,:,:].T))
                iter_sspw.append(metrics().sum_sq_port_wts(wts[m,:,:].T))
            
            
            stats[names[m]] += [np.mean(iter_turnover),np.std(iter_turnover)/n_sims**0.5,
                 np.mean(iter_sspw),np.std(iter_sspw)/n_sims**0.5]
            
            
        col_names = ['Avg Turnover', 'Avg Turnover (se)','Avg SSPW','Avg SSPW (se)']
        
        df = pd.DataFrame.from_dict(stats,orient = 'index',columns = col_names)
        df.index.name = 'Model'
        
        print(tabulate(df,headers='keys', tablefmt='psql'))
        
        if return_df:
            return df
    
