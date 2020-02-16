import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

class gHRP:
    
    def __init__(self,correlator = np.corrcoef, dmetric = 'euclidean', linkage_type = 'single'):
        self.corr = correlator
        self.dmetric = dmetric
        self.linkage_type = linkage_type
    
    def _tcluster(self,data):
        '''
        Parameters
            data: NxT ndarray with N stocks and T returns
        Returns:
            link: linkage ndarray. See linkage function in sciypy for details
        '''
        corr = self.corr(data)
        dmat = np.sqrt(0.5*(1-corr))
        
        link = linkage(pdist(dmat,metric=self.dmetric),method=self.linkage_type)
        
        return link
        

if __name__ == '__main__':
    
    print('**** Running Tests for Models: ****')
    
    
    