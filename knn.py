import torch
from math import ceil

class CudaKNN:

    '''
    k-NN PyTorch Implementation
    '''
    
    def __init__(self, k, p=2, cuda=False):

        '''
        Constructor:

        Inputs:
        - k: Number of neighbors
        - p: Norm parameter (p-norm)
        - cuda: if true use gpu for computation
        '''
        
        self.k = k
        self.p = p
        if cuda:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                raise AttributeError('`cuda` parameter was set to True however no available cuda device was found.')
        else:
            self.device = 'cpu'
        self.X = None
        self.Y = None
        
    def fit(self, X, Y):

        '''
        fitting k-NN to data. The model is the data

        Inputs:
        - X: data (torch.Tensor)
        - Y: labels (torch.Tensor)
        '''
        
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)

    def pointwise_distances(self, X_train, X, dim_):

        '''
        Computes pointwise p-norm between input(s) and training samples X_train

        Inputs:
        - X_train: training samples
        - X : inputs
        - dim_ : dimension to reduce

        Outputs:
        - pointwise p-norm
        '''

        return torch.norm(X_train-X, p=self.p, dim=dim_)

    def check_input(self, X):

        '''
        Check the shape of the input and reformat it if necessary

        Input:
        - X: input

        Output:
        - X: reformated input
        - dim_: dimension to reduce when computing the norm
        '''

        if len(X.shape) == 3:
            if X.shape[1] != 1:
                raise TypeError(f'X must be 1D or 2D or 3D with dimensionality 1 along axis 1')
            else:
                dim_ = 2
        elif len(X.shape) == 2:
            if 1 in X.shape:
                dim_ = 1
            else:
                X = X.unsqueeze(1)
                dim_ = 2
        elif len(X.shape) == 1:
            X = X.reshape(1, -1)
            dim_ = 1
        else:
            raise TypeError(f'X must be 1D or 2D or 3D with dimensionality 1 along axis 1')

        X = X.to(self.device)

        return X, dim_

    def predict(self, X):

        '''
        Predict the labels of the inputs X

        Input:
        - X: Input

        Output:
        - predicted labels
        '''

        X, dim_ = self.check_input(X)
        distances = self.pointwise_distances(self.X, X, dim_)

        if dim_ == 1:
            sorted_indices = torch.argsort(distances, dim=0)
            k_indices = sorted_indices[:self.k]
            return torch.mode(self.Y[k_indices], dim=0)[0]

        else:
            sorted_indices = torch.argsort(distances, dim=1)
            k_indices = sorted_indices[:,:self.k]
            return torch.mode(self.Y[k_indices], dim=1)[0]
