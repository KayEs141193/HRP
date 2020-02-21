import model,data
import pandas as pd
import numpy as np

def data_gen():
    np.random.randn(5,10)

def simulate(data,model_list,name,iterations=100000,test_split=5,step=2,mode='train'):
    '''
        Produces analytics and test samples for models based on the passed data
        
        Parameters
            data: Data Generator object[The passed data of TxN format, where T is number of days and N is number of assets]
            model_list: List of portfolio objects
            name: Name of models being sent 
            iterations: Number of MC iterations 
        Returns:
            Initial portfolio allocations, Average returns, Returns Variance     
    '''
    numStocks = data.generate().shape[1]
    if mode == 'train':
        initWeights = {name[i]:np.zeros(numStocks) for i in range(len(model_list))}
        cumRet = {name[i]:0 for i in range(len(model_list))}
        cumVol = {name[i]:0 for i in range(len(model_list))}
        for iters in range(iterations):
            train_data = data.generate()#pd.DataFrame(np.random.randn(10,5))
            inSample = train_data[:test_split]#train_data.iloc[:test_split]
            outSample = train_data[test_split:]#train_data.iloc[test_split:]
            
            for i,model in enumerate(model_list):
                dailyRet = []
                weights = []

                # Computing average initial weights for the model
                initWeights[name[i]] += model.allocate(inSample.T).values
            
                # Testing for model returns 
                for period in range(0,outSample.shape[0],step):
                    if period:
                        w = model.allocate(train_data[:period].T).values
                    else:
                        w = initWeights[name[i]]

                    dailyRet.extend(np.dot(w,outSample[period:period+step].T))
                    weights.append(w)
                
                cumRet[name[i]] = np.mean(np.array(dailyRet))
                cumVol[name[i]] = np.var(np.array(dailyRet))

        allocations = pd.DataFrame(initWeights)/iterations
        return allocations,cumRet,cumVol,np.array(weights)
        
    if mode == 'test':
        pass
        # For Testing with real data

