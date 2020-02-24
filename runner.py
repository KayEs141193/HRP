import simulation
from data import LopezGenerator, GaussianGenerator, DynamicGenerator
from model import gHRP, gIVP, gCLA
from util import plotUtil
import numpy as np

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
    
def run_downturn_scenarios():
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
    n = 10
    n_iter = 100
    
    linkage = 'single'
    
    sigma = np.ones(n)*0.01
    mean  = np.zeros(n)
    
    g1 = GaussianGenerator(sigma,mean,'Diagnol',n)
    g2 = GaussianGenerator(sigma*2,mean,'Correlated',n,(0.9,1))
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    datagen = DynamicGenerator(np.array([282,132,146]),[g1,g2,g1],n)
    res = simulation.simulateAll(datagen,[m,m2,m3],22*14,260,22,n_iter)
    
    plotUtil.plot_wts_timeseries(res,['HRP','IVP','CLA'])
    plotUtil.gen_summary_statistics(res,['HRP','IVP','CLA'])
    
def run_lopez_replication():
    params = {  'nObs': 520,
                'sLength':260,
                'size0':5,
                'size1':5,
                'mu0':0,
                'sigma0':.01,
                'sigma1F':0.25}
    
    n_iter = 100
    linkage = 'complete'
    
    m = gHRP(linkage_type=linkage)
    m2 = gIVP()
    m3 = gCLA()
    datagen = LopezGenerator(params['sLength'],params['size0'],params['size1'],params['mu0'],params['sigma0'],params['sigma1F'])
    res = simulation.simulateAll(datagen,[m,m2,m3],22*12,260,22,n_iter)
    
    plotUtil.plot_wts_timeseries(res,['HRP','IVP','CLA'])
    plotUtil.gen_summary_statistics(res,['HRP','IVP','CLA'])