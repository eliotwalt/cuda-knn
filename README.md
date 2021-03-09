# Cuda-KNN

K nearest neighbors cuda-accelerated implementation using `PyTorch`. Usage example can be found in `mnist.ipynb` which applies the model to mnist dataset.

## Usage

Start by installing the requirements in `requirements.txt`. 
To apply the model to a new dataset, first write some code to load your data in the `load_data` function in `utils.py`. Give a name to your dataset and wrap your code in a `if dataset == $dataset_name`. Then, the `eval.py` script allows to evaluate the model on fixed parameters and save a `PCA` and a `CudaKNN` model. CLI arguments:
```bash
python eval.py -h
usage: eval.py [-h] --dataset DATASET --test_size TEST_SIZE --val_size VAL_SIZE --batch_size BATCH_SIZE
               [--flatten] [--cuda] --pca_components PCA_COMPONENTS --neighbors NEIGHBORS --p_norm P_NORM
               [--verbose] [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        name of dataset
  --test_size TEST_SIZE, -ts TEST_SIZE
                        number of test samples
  --val_size VAL_SIZE, -vs VAL_SIZE
                        number of validation samples
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        batch size
  --flatten, -f         flatten data if flag specified
  --cuda, -c            use cuda if flag specified
  --pca_components PCA_COMPONENTS, -n PCA_COMPONENTS
                        number of components for PCA
  --neighbors NEIGHBORS, -k NEIGHBORS
                        number of neighbors for knn
  --p_norm P_NORM, -p P_NORM
                        norm parameter
  --verbose, -v         verbose activated if flag specified
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
```
Example for mnist (11Gb of gpu memory):
```bash
python eval.py --dataset mnist \
               --test_size 10000 \
               --val_size 5000 \
               --batch_size 64 \
               --flatten \
               --cuda \
               --pca_components 100 \
               --neighbors 3 \
               --p_norm 2 \
               --verbose \
               --output_dir model
```