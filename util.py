# Takes as input the model allocations, mean returns and mean variance of each portfolio, outputs 2 reports and a chart


import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, binned_statistic
from tabulate import tabulate
import os

import matplotlib.pyplot as plt


class metrics():
    
    def __init__(self):
        self.rf = 0
     
    
    def calc_rets(self,data):
        
        '''
        Paramters:
            data: Takes T x N np array of returns of N assets across T timeperiods
            
        Returns:
            Sharpe ratio for portfolio returns for the time period
        '''
        
        ret_data = 1 + data
        
        tot_ret = np.prod(ret_data,axis = 0) - 1
        
        tot_log_ret = np.sum(data,axis = 0)
        
        avg_daily_ret = np.mean(data,axis = 0)
        
        avg_log_ret = np.mean(data,axis = 0)
        
        return tot_ret, tot_log_ret, avg_daily_ret, avg_log_ret
    

    
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
    def plot_wts_timeseries(self,res,names,save_output = False):
        
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
            plt.figure(figsize = (16,10))
            plt.plot(avg_wts[m].T)
            
            model_name = 'Model - ' + names[m]
            plt.title(model_name)
            plt.xlabel('Rebalancing period')
            plt.ylabel('Asset weights')
            plt.ylim(0.0,1.0)
            if save_output:
                if not os.path.exists('./runner_outputs'):
                    os.makedirs('./runner_outputs')
                path = './runner_outputs/' + model_name + '.png'
                plt.savefig(path)
            plt.show()

        
            
            
    
    @classmethod
    def gen_summary_statistics(self,res,names,return_df = False, save_output = False):
    
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
                
        stats_means = {model : [] for model in names }
        stats_se = {model : [] for model in names }
        
        for m in range(M):
            iter_turnover = []
            iter_sspw = []
            iter_tot_ret = []
            iter_tot_log_ret = []
            iter_avg_daily_ret = []
            iter_avg_daily_log_ret = []
            
            
            for i in range(n_sims):
                
                wts, dailyrets = res[i]
                iter_turnover.append(metrics().avg_turnover(wts[m,:,:].T))
                iter_sspw.append(metrics().sum_sq_port_wts(wts[m,:,:].T))
                
                tot_ret, tot_log_ret, avg_daily_ret, avg_log_ret = metrics().calc_rets(dailyrets[m].T)
                
                iter_tot_ret.append(tot_ret)
                iter_tot_log_ret.append(tot_log_ret)
                iter_avg_daily_ret.append(avg_daily_ret)
                iter_avg_daily_log_ret.append(avg_log_ret)
                
            
            stats_means[names[m]] += [np.mean(iter_turnover),
                 np.mean(iter_sspw),
                 np.mean(iter_tot_ret), 
                 np.mean(iter_tot_log_ret),
                 np.mean(iter_avg_daily_ret),
                 np.mean(iter_avg_daily_log_ret)
                 ]
            
            stats_se[names[m]] += [np.std(iter_turnover)/n_sims**0.5,
                 np.std(iter_sspw)/n_sims**0.5,
                 
                 np.var(iter_tot_ret,ddof=1),
                 np.var(iter_tot_log_ret,ddof=1),
                 np.var(iter_avg_daily_ret,ddof=1),
                 np.var(iter_avg_daily_log_ret,ddof=1)
                 ]
            
            
            
        col_names_means = ['Avg Turnover', 'Avg SSPW',
                     'Avg Total Return',
                     'Avg Total Log Return',
                     'Avg Daily Return',
                     'Avg Daily Log Return'
                     ]
        
        col_names_se = ['Avg Turnover (se)','Avg SSPW (se)',
                        'Var Total Return',
                        'Var Total Log Return',
                        'Var Daily Return',
                        'Var Daily Log Return'
                     ]
        
        df_means = pd.DataFrame.from_dict(stats_means,orient = 'index',columns = col_names_means)
        df_means.index.name = 'Model'
        
        df_se = pd.DataFrame.from_dict(stats_se,orient = 'index',columns = col_names_se)
        df_se.index.name = 'Model'
        
        print(tabulate(df_means,headers='keys', tablefmt='psql'))
        print(tabulate(df_se,headers='keys', tablefmt='psql'))
        
        if return_df:
            return df_means,df_se
        
        if save_output:

            if not os.path.exists('./runner_outputs'):
                os.makedirs('./runner_outputs')
        
            filename_means_path =  './runner_outputs/' + str(names[0][3:]) + '_means.csv'
            filename_std_path =  './runner_outputs/' + str(names[0][3:]) + '_std.csv'
        
            df_means.to_csv(filename_means_path)
            df_se.to_csv(filename_std_path)
    

    