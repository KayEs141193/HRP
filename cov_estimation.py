from sklearn.covariance import EmpiricalCovariance,ShrunkCovariance,ledoit_wolf,GraphicalLasso
import numpy as np

class Covariance():
    
    def __init__(self,data):
        '''
        Assuming data has an input format of N stocksXT samples
        '''
        self.data = data.transpose()

    def cov(self,decay=0.01):
        # aweights is the weights assigned
        aweights = np.arange(self.data.shape[0])
        aweights = np.exp(-decay*aweights)
        cov_mat = np.cov(self.data.transpose(),aweights=aweights)
        print(cov_mat.shape)
        return cov_mat 

    def corr_from_cov(self,cov_mat):
        d = (np.identity(cov_mat.shape[0])*cov_mat)**0.5
        d_1 = np.linalg.inv(d)
        return d_1.dot(cov_mat).dot(d_1)

    def corr(self):
        cov_mat = self.cov()
        return self.corr_from_cov(cov_mat)

    def shrunk_corr(self,alpha = 0.3,gamma = 0.01,lwolf=False): 
        cov_mat = self.shrunk_cov(alpha,gamma,lwolf)
        return self.corr_from_cov(cov_mat)

    def graphical_lasso_corr(self,lam=0.01):
        cov_mat = self.graphical_lasso_cov(lam)
        return self.corr_from_cov(cov_mat)

    def shrunk_cov(self,alpha = 0.2,gamma = 0.01,lwolf=False):
        if lwolf:
            cMat = ledoit_wolf().fit(self.data)
            return cMat.covariance_
        else: 
            cov_mat = self.cov()
            cMat = alpha*cov_mat + (1-alpha)*np.identity(cov_mat.shape[0])*cov_mat
            return cMat

    def graphical_lasso_cov(self,lam=0.01):
        cMat = GraphicalLasso(alpha = lam).fit(self.data)
        return cMat.covariance_

