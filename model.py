import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from simulation import *

class gHRP:
    
    def __init__(self,correlator = np.corrcoef, dmetric = 'euclidean', linkage_type = 'single'):
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
        corr = self.corr(data)
        dmat = np.sqrt(0.5*(1-corr))
        
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
        cov = np.cov(data)
        
        sortIx = self._getQuasiDiag(link)
        w = self._getRecBipart(cov,sortIx)
        
        return w
        
        
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
    
    #print(hrp.allocate(T1))



    
    