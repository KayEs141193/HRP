import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from pypfopt.efficient_frontier import EfficientFrontier

class Correlators:
    '''
    Base correlator class
    '''
    def corr(self, data):
        covmat = self.cov(data)
        std  = np.sqrt(covmat[np.arange(covmat.shape[0]),np.arange(covmat.shape[0])])
        return (covmat/std.reshape(-1,1))/std.reshape(1,-1)
    
class PearsonCorrelator(Correlators):
    '''
    Pearson Correlator
    '''
    def __init__(self,decay=0):
        self.decay = decay
    
    def cov(self, data):
        data = data.T
        aweights = np.arange(data.shape[0])
        aweights = np.exp(-self.decay*aweights)
        cov_mat = np.cov(data.T,aweights=aweights)
        return cov_mat

class gHRP:
    
    def __init__(self,correlator = PearsonCorrelator(), dmetric = 'euclidean', linkage_type = 'single'):
        self.corr = correlator
        self.dmetric = dmetric
        self.linkage_type = linkage_type
    
    def _tcluster(self,data):
        '''
        Perform hierarchical clustering on stocks by first creating a distance mmatrix based on the
        correlation matrix
        
        Parameters
            data: NxT ndarray with N stocks and T returns
        Returns:
            link: linkage ndarray. See linkage function in sciypy for details
        '''
        
        epsilon = 0.000000001
        corr = self.corr.corr(data)
        dmat = np.sqrt(0.5*(1+epsilon-corr))
        link = linkage(pdist(dmat,metric=self.dmetric),method=self.linkage_type)
        
        return link
        
    def _getQuasiDiag(self,link):
        '''
        Reorders the covariance matrix to place similar stocks closer together and dissimilar stocks
        far apart.
        
        Parameters:
            link: Link matrix produced by hierarchical clustering
        Returns
            order: ndarray containing ordering of the stocks based on the hierarchical clustering
        '''
        
        link = link.astype(int)
        sortIx = pd.Series([link[-1,0],link[-1,1]])
        numItems = link[-1,3]
        
        while sortIx.max() >= numItems: 
            # Run as long as clusters is unresolved exist in sortIx
            
            sortIx.index = range(0,sortIx.shape[0]*2,2) # re-indexing to ensure cluster items fall next to each other
            df0 = sortIx[ sortIx >= numItems ] # Get clusters
            
            # Each entry in the link matrix is a cluster, therefore we reset from 0 to index the correct cluster
            i, j = df0.index, df0.values-numItems
            sortIx[i] = link[j,0] # Replacing clusters with their item1
            
            df0 = pd.Series(link[j,1], index=i+1) # index item2 properly to ensure cluster elements fall close together
            
            sortIx = sortIx.append(df0) # Append item2
            sortIx = sortIx.sort_index() # re-sort
            sortIx.index = range(sortIx.shape[0]) # re-index
        
        return sortIx.to_numpy()
    
    
    def _getClusterVar(self,cov,items):
        V = cov[items,:][:,items]
        dinv = (np.eye(len(items))*(1/V))[range(len(items)),range(len(items))]
        
        w = dinv/dinv.sum()
        
        return w.dot( V.dot(w) )
    
    def _getRecBipart(self,cov,sortIx):
        '''
        Perform weight allocation given covariance matrix and ordered stocks
        
        Parameters:
            cov: covariance matrix
            sortIx: ordered stocks
        Returns:
            w: pandas Series weight allocation per stock, #stock given in the index
            
        '''
        
        w = pd.Series(1,index=sortIx) # Initialize the weights to 1
        cItems = [sortIx] # initialize all items in one cluster
        
        while len(cItems) > 0:

            cItems = [ i[j:k] for i in cItems for j,k in ((0,int(len(i)/2)),(int(len(i)/2),len(i))) if len(i) > 1 ] # bi-section
            # Note: single items are removed, this reduces the size of citems eventually
            
            for i in range(0,len(cItems),2):
                
                cItems0 = cItems[i] # cluster 1
                cItems1 = cItems[i+1] # cluster 2
                
                cVar0 = self._getClusterVar(cov,cItems0)
                cVar1 = self._getClusterVar(cov,cItems1)
                
                alpha = 1 - cVar0/(cVar0+cVar1)
                
                w[cItems0] *= alpha
                w[cItems1] *= 1-alpha
                
        return w
    
    def allocate(self,data):
        '''
        Perform HRP-based weight allocation
        
        Parameters
            data: NxT ndarray with N stocks and T returns
        Returns:
            w: pandas Series weight allocation per stock, #stock given in the index
        '''
        
        link = self._tcluster(data)
        
        # Covariance calculated here is different from self.correlate. POTENTIAL INCONSISTENCY
        cov = self.corr.cov(data)
        
        sortIx = self._getQuasiDiag(link)
        w = self._getRecBipart(cov,sortIx)
        
        return w

class gIVP:
    
    def __init__(self,cov = PearsonCorrelator()):
        self.cov = cov


    def allocate(self,data):
        '''
        Allocates a weight in inverse proportion to the subset’s variance.
        
        Parameters:
            data: NxT ndarray with N stocks and T returns
            
        Returns:
            w: pandas Series weight allocation per stock, #stock given in the index

        '''
        _cov = self.cov.cov(data)
        _ivp = 1./np.diag(_cov)
        _ivp /= _ivp.sum()

        
        return pd.Series(_ivp)
    
    
class gEFO:
    
    def __init__(self,cov = PearsonCorrelator()):
        self.cov = cov
        
    def allocate(self,data):
        '''
        Compute CLA's minimum variance portfolio
        
        Parameters:
            data: NxT ndarray with N stocks and T returns
            
        Returns:
            w: pandas Series weight allocation per stock, #stock given in the index
        '''
        
        _cov = (self.cov.cov(data))
        _mu=np.mean(data,axis = 1)
        
        # Optimise for maximal Sharpe ratio
        ef = EfficientFrontier(_mu, _cov)
        w = ef.max_sharpe()

        return pd.Series(w)
            
 
        
if __name__ == '__main__':
    
    print('**** Running Tests for Models: ****')
    
    hrp = gHRP()
    
    T1 = np.array([[1,2,3,4,5,6],[6,5,4,3,2,1],[4,1,4,3,4,-4],[1,2,1,2,1,2]])
    R1 = np.array([[1.        , 2.        , 0.69821736, 2.        ],
                   [0.        , 3.        , 0.86454545, 2.        ],
                   [4.        , 5.        , 1.28509114, 4.        ]])
    V1 = hrp._tcluster(T1)
    #assert( hrp._tcluster(T1) == R1 )
    
    V2 = hrp._getQuasiDiag(V1)
    
    print(hrp.allocate(T1))



    
    