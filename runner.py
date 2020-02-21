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
    Generates basic scenarios with varying linkages.
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Exponential weight decay: 0.05
        3. Correlator: Basic correlation
    '''
    pass
    
def generate_downturn_scenarios(n,t1,t2,t3):
    '''
    Generates basic scenarios with varying linkages.
    Fixed:
        1. Rebalancing Parameter (l): 22
        2. Exponential weight decay: 0.
        3. Correlator: Basic correlation
        4. Underlying return generating process: Dynamic Generator
    Parameters:
        t1: initial period of normal return
        t2: period of downturn
        t3: final period of normal return
    '''
    pass
