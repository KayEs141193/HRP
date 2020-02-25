import simulation
from data import LopezGenerator, GaussianGenerator, DynamicGenerator, TGenerator, SkewedNormGenerator, RealDataGenerator
from model import gHRP, gIVP, gCLA
from util import plotUtil
import numpy as np
import pandas as pd

'''
    Adding different scenarios to run
    
    Scenario Description:
        1. Name
        2. Parameter Details
        3. Generator

    Synthetic Scenarios
    Things you can vary:
        1. No.of assets
        2. Time horizon of analysis
        3. Underlying return generating process i.e generator
        4. Rebalancing parameter
        5. Exponential weight-decay parameter
        6. Correlators
        7. Linkages for HRP
'''

def generate_basic_scenarios(n,t,corrp):
    '''
    Generates basic scenarios with varying linkages with Lopez generator
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Exponential weight decay: 0.05
        3. Correlator: Basic correlation
    '''
    pass
    
def run_downturn_scenarios(n_iter,sigma,factor,linkage,n,rholim):
    '''
    Generates basic scenarios with varying linkages.
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Window = 260.
        3. Correlator: Basic correlation
        4. Underlying return generating process: Dynamic Generator
    Parameters:
        t1: initial period of normal return
        t2: period of downturn
        t3: final period of normal return
    '''
    
    sigma = np.ones(n)*sigma
    mean  = np.zeros(n)
    
    g1 = GaussianGenerator(sigma,mean,'Diagnol',n)
    g2 = GaussianGenerator(sigma*factor,mean,'Correlated',n,rholim)
    
    m = [ gHRP(linkage_type=link) for link in ['single','complete','average','centroid','median','ward'] ]
    
    m2 = gIVP()
    m3 = gCLA()
    
    getMeta = [1,4,10,19,25,40]
    names = [ name+'_downturn__iter_'+str(n_iter)+'__linkage_'+linkage for name in [*[ 'HRP_'+name+'_' for name in  ['single','complete','average','centroid','median','ward']],'IVP','CLA']]
    
    datagen = DynamicGenerator(np.array([282,132,150]),[g1,g2,g1],n)
    res = simulation.simulateAll(datagen,m+[m2,m3],22*50,260,22,n_iter,getMeta,names)    
    
    plotUtil.plot_wts_timeseries(res,names,save_output=True)
    plotUtil.gen_summary_statistics(res,names,save_output=True)


def run_skewedNorm_scenarios(n_iter,sigma,linkage,n,a):
    '''
    Generates basic scenarios with varying linkages.
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Window = 260.
        3. Correlator: Basic correlation
        4. Underlying return generating process: Dynamic Generator
    Parameters:
        t1: initial period of normal return
        t2: period of downturn
        t3: final period of normal return
    '''
    
    sigma = np.ones(n)*sigma
    mean  = np.zeros(n)
    
    datagen = SkewedNormGenerator(a*np.ones(n),sigma,mean,'CorrelatedGroups',np.array([n//2,n-n//2]),[(0.8,1.0),(0.2,0.4)])
    
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    
    res = simulation.simulateAll(datagen,[m,m2,m3],22*14,260,22,n_iter)
    
    names = [ name+'_skewedNorm__iter_'+str(n_iter)+'__linkage_'+linkage for name in ['HRP','IVP','CLA']]
    
    plotUtil.plot_wts_timeseries(res,names,save_output=True)
    plotUtil.gen_summary_statistics(res,names,save_output=True)

def run_T_scenarios(n_iter,sigma,linkage,n,df):
    '''
    Generates basic scenarios with varying linkages.
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Window = 260.
        3. Correlator: Basic correlation
        4. Underlying return generating process: Dynamic Generator
    Parameters:
        t1: initial period of normal return
        t2: period of downturn
        t3: final period of normal return
    '''
    
    sigma = np.ones(n)*sigma
    mean  = np.zeros(n)
    
    datagen = TGenerator(df*np.ones(n),sigma,mean,'CorrelatedGroups',np.array([n//2,n-n//2]),[(0.8,1.0),(0.2,0.4)])
    
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    
    res = simulation.simulateAll(datagen,[m,m2,m3],22*14,260,22,n_iter)
    
    names = [ name+'_T__iter_'+str(n_iter)+'__linkage_'+linkage for name in ['HRP','IVP','CLA']]
    
    plotUtil.plot_wts_timeseries(res,names,save_output=True)
    plotUtil.gen_summary_statistics(res,names,save_output=True)
    

def run_lopez_replication(n_iter,linkage):
    params = {  'nObs': 520,
                'sLength':260,
                'size0':5,
                'size1':5,
                'mu0':0,
                'sigma0':.01,
                'sigma1F':0.25}
        
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    
    datagen = LopezGenerator(params['sLength'],params['size0'],params['size1'],params['mu0'],params['sigma0'],params['sigma1F'])
    res = simulation.simulateAll(datagen,[m,m2,m3],22*12,260,22,n_iter)
    
    names = [ name+'__iter_'+str(n_iter)+'__linkage_'+linkage for name in ['HRP','IVP','CLA']]
    
    plotUtil.plot_wts_timeseries(res,names,save_output=True)
    plotUtil.gen_summary_statistics(res,names,save_output=True)


def run_real_world(linkage,datatype='S&P'):
    
    n = 1 # This parameter doesn't matter
    n_iter = 1 # Changing this is useless as same data will be returned
    
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    
    datagen = RealDataGenerator(datatype)
    
    if datatype == 'S&P':
        res = simulation.simulateAll(datagen,[m,m2,m3],4792,260,22,n_iter) # Use for S&P
        legends = ['Materials	','Industrials'	,'Consumer Disc',	'Consumer Staples', 'Energy',	'Financial',	'Utilities',	'HealthCare',	'Tech']
    else:
        res = simulation.simulateAll(datagen,[m,m2,m3],1165,260,22,n_iter)
        legends = ['Reits',	'Russell3k'	,'SP500'	,'FTSE AW ex US',	'MSCI EM	',	'MBS ETF',	'Intl Bond','EMD Local FI',	'Junk Bonds',	'GOLD']
        
    names = [ name+ '__' + datatype +'__linkage_'+linkage for name in ['HRP','IVP','CLA']]
    
    plotUtil.plot_wts_timeseries(res,names,asset_legends=legends,save_output=True)
    plotUtil.gen_summary_statistics(res,names,save_output=True)
    
    for i,name in enumerate(names):
        pd.DataFrame(res[0][0][i,:,:]).to_csv('./runner_outputs/weights_'+name+'_'+datatype+'_'+linkage+'.csv')
        pd.Series(res[0][1][i,:]).to_csv('./runner_outputs/ptfDailyRets_'+name+'_'+datatype+'_'+linkage+'.csv')


#for link in ['single','complete','average','centroid','median','ward']:
run_downturn_scenarios(1,0.01,5,'single',10,(0.8,1.0))