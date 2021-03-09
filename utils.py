import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import matplotlib.pyplot as plt
from knn import CudaKNN
from time import time

def load_data(test_size, validation_size, dataset, flatten=True):

    '''
    Load mnist into torch tensors and perform cross validation split

    Inputs:
    - test_size: number or fraction of samples to assign to the test set
    - validation_size: number or fraction of training samples to assign to the validation set
    - flatten: if True X's are of shape (1, 1, 28, 28) otherwise (784, 1)

    Ouptuts:
    - train_input, train_target: training set
    - val_input, val_target: validation set
    - test_input, test_target: test set
    '''
    
    if dataset == 'mnist':
        X, Y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])

        X = X[permutation]
        Y = Y[permutation]
        X = X.reshape((X.shape[0], -1))
        train_input, test_input, train_target, test_target = train_test_split(
            X, Y, test_size=test_size
        )
        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=validation_size
        )

        train_input = torch.from_numpy(train_input).reshape(-1,1,28,28)
        train_target = torch.from_numpy(train_target.astype(np.int8))
        val_input = torch.from_numpy(val_input).reshape(-1,1,28,28)
        val_target = torch.from_numpy(val_target.astype(np.int8))
        test_input = torch.from_numpy(test_input).reshape(-1,1,28,28)
        test_target = torch.from_numpy(test_target.astype(np.int8))
    
    # Add custom loader here

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)
        val_input = val_input.clone().reshape(val_input.size(0), -1)
        
    return train_input, train_target, val_input, val_target, test_input, test_target

def show_pairwise_projection(X, Y):

    '''
    scatter plot of the data X projected on all pairs of dimensions

    Inputs:
    - X: data
    - Y: label (used for color)
    '''
    
    f, axs = plt.subplots(X.shape[1], X.shape[1])
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for y in torch.unique(Y):
                XX = X[Y==y]
                axs[i,j].scatter(XX[:,i], XX[:,j], alpha=.25)
                axs[i,j].tick_params(left=False,
                                     bottom=False,
                                     labelleft=False,
                                     labelbottom=False)
    
    plt.show()

def pca_cudaknn_grid_search(X_train, Y_train, X_val, Y_val,
                            ns_components, ns_neighbors, 
                            batch_size, p, cuda,
                            verbose=True):

    '''
    Hyper-parameter grid search for PCA + CudaKNN

    Inputs:
    - X_train, Y_train: Train set
    - X_val, Y_val: Validation set
    - ns_components: list of number of components to test
    - ns_neighbors: list of number of neighbors to test
    - batch_size: size of the prediction batches. Can be an Integer or a list of integers of
                  the same length as ns_components. In this case, batch_size[i] is used with
                  ns_components[i].
    - p: norm parameter (p-norm)
    - cuda: if true use gpu for computation
    - verbose: if false, no outputs in console
    '''

    errors = {n: None for n in ns_components}
    Y_pred = {n: {k: None for k in ns_neighbors} for n in ns_components}
    times = {n: {k: None for k in ns_neighbors} for n in ns_components}

    if hasattr(batch_size, '__iter__'):
        if len(batch_size) != len(ns_components):
            raise AttributeError(f'If batch_size is an iterable it must have the same length as ns_components')
        else:
            batch_is_list = True
    # Standardize for PCA
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    for i, n in enumerate(ns_components):
        # Apply PCA
        if verbose:
            print(f'[PCA n={n}]')
        pca = PCA(n_components=n).fit(X_train_std)
        X_train_pca = torch.from_numpy(pca.transform(X_train))
        X_val_pca = torch.from_numpy(pca.transform(X_val))
        n_errors = []
        if batch_is_list:
            _batch_size = batch_size[i]
        else:
            _batch_size = batch_size

        for k in ns_neighbors:
            # Instantiate CudaKNN
            knn = CudaKNN(k=k, p=p, cuda=cuda)
            knn.fit(X_train_pca, Y_train)
            y_preds = []
            error = 0
            nk_times = []

            for i in range(_batch_size, Y_val.shape[0]+_batch_size, _batch_size):
                # Predict batch
                y_val = Y_val[i-_batch_size:i]
                x_val = X_val_pca[i-_batch_size:i]
                t0 = time()
                y_pred = knn.predict(x_val).cpu()
                t1 = time()
                error += (y_pred!=y_val).int().sum().item()
                y_preds.append(y_pred)
                nk_times.append(t1-t0)
                j = min(Y_val.shape[0], i+1)
                if verbose:
                    print(f'[k-NN k={k}] ({j}/{Y_val.shape[0]}) error rate = {error/(j)*100:.4f}%', end='\r')

            # Collect results
            if verbose:
                print()
            error /= Y_val.shape[0]
            n_errors.append(error)
            y_preds = torch.cat(y_preds, dim=0)
            Y_pred[n][k] = y_preds
            times[n][k] = nk_times
            if verbose:
                print(f'[k-NN k={k}] error rate = {error*100:.4f}%')

        errors[n] = n_errors
        if verbose:
            print()
    
    return errors, Y_pred, times

def evaluate_pca_knn(X_train, Y_train, X_val, Y_val, 
                     pca_components, neighbors, batch_size, 
                     p, cuda, verbose=True):

    '''
    evaluate pca+cudaknn pipeline

    Inputs:
    - X_train, Y_train: Train set
    - X_val, Y_val: Validation set
    - pca_components: number of pca components
    - neighbors: number of neighbors
    - batch_size: size of the prediction batches.
    - p: norm parameter (p-norm)
    - cuda: if true use gpu for computation
    - verbose: if false, no outputs in console
    '''

    error = 0
    compute_time = 0

    # Standardize for PCA
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    # PCA
    pca = PCA(n_components=pca_components).fit(X_train_std)
    X_train_pca = torch.from_numpy(pca.transform(X_train))
    X_val_pca = torch.from_numpy(pca.transform(X_val))

    knn = CudaKNN(k=neighbors, p=p, cuda=cuda)
    knn.fit(X_train_pca, Y_train)

    if verbose:
        print()
    niter = 0
    for i in range(batch_size, Y_val.shape[0]+batch_size, batch_size):
        niter += 1
        # Predict batch
        y_val = Y_val[i-batch_size:i]
        x_val = X_val_pca[i-batch_size:i]
        t0 = time()
        y_pred = knn.predict(x_val).cpu()
        t1 = time()
        error += (y_pred!=y_val).int().sum().item()
        compute_time += t1-t0
        j = min(Y_val.shape[0], i+1)
        if verbose:
            print(f'[k-NN k={neighbors}] ({j}/{Y_val.shape[0]}) error rate = {error/(j)*100:.4f}%', end='\r')
    if verbose:
        print()
    error /= Y_val.shape[0]
    
    return pca, knn, error, compute_time, compute_time/niter 
