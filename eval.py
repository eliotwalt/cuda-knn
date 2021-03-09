import argparse
from utils import load_data, evaluate_pca_knn
import os
from datetime import datetime
import pickle

def directory(x):

    if os.path.isdir(x):
        return x
    raise argparse.ArgumentTypeError(f'{x} must be a directory')

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='name of dataset')
    parser.add_argument('--test_size', '-ts', type=int, required=True,
                        help='number of test samples')
    parser.add_argument('--val_size', '-vs', type=int, required=True,
                        help='number of validation samples')
    parser.add_argument('--batch_size', '-bs', type=int, required=True,
                        help='batch size')
    parser.add_argument('--flatten', '-f', action='store_true', required=False,
                        help='flatten data if flag specified')
    parser.add_argument('--cuda', '-c', action='store_true', required=False,
                        help='use cuda if flag specified')
    parser.add_argument('--pca_components', '-n', type=int, required=True,
                        help='number of components for PCA')
    parser.add_argument('--neighbors', '-k', type=int, required=True,
                        help='number of neighbors for knn')
    parser.add_argument('--p_norm', '-p', type=int, required=True,
                        help='norm parameter')
    parser.add_argument('--verbose', '-v', action='store_true', required=False,
                        help='verbose activated if flag specified')
    parser.add_argument('--output_dir', '-o', type=directory)

    return parser.parse_args()

def main():

    now = datetime.now()
    now = now.strftime("%d_%m_%y-%H%M")

    args = get_args()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(test_size=args.test_size,
                                                               validation_size=args.val_size,
                                                               flatten=args.flatten,
                                                               dataset=args.dataset)
    pca, cudaknn, error, compute_time, mean_compute_time = evaluate_pca_knn(X_train=X_train, Y_train=Y_train,
                                                                X_val=X_val, Y_val=Y_val,
                                                                pca_components=args.pca_components,
                                                                neighbors=args.neighbors,
                                                                batch_size=args.batch_size,
                                                                p=args.p_norm, cuda=args.cuda,
                                                                verbose=args.verbose)
    
    pca_path = os.path.join(args.output_dir, f'pca_{now}.pkl')
    cudaknn_path = os.path.join(args.output_dir, f'cudaknn_{now}.pkl')
    with open(pca_path, 'wb') as h:
        pickle.dump(pca, h)
    with open(cudaknn_path, 'wb') as h:
        pickle.dump(cudaknn, h)
    
    print()
    print('Validation Results:')
    print(f'Mean error rate: {error*100:.4f}%')
    print(f'Total evaluation time: {compute_time:.4f}[s]')
    print(f'Mean batch evaluation time: {mean_compute_time:.4f}[s]')
    print(f'pca object at: {pca_path}')
    print(f'cudaknn object at: {cudaknn_path}')

if __name__ == '__main__':

    main()
