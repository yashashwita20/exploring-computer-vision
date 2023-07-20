<!-- #region -->
# Image Classification with Machine Learning


## Overview

This repository includes 5 parts:

1. KNN
2. Linear Regression
3. Logistic Regression
4. Neural Network in NumPy
5. Classification using Neural Network

File structure:

```
Image Classification with Machine Learning
├── README.md  (this file)
├── ece285  (source code folder)
├── *.ipynb  (notebooks)
├── get_datasets.py  (download script)
├── algorithms  (algorithm implementation)
├── layers  (neural net layers)
├── utils (utility scripts)
└── datasets  (datasets folder)
```

## Prepare Datasets

Before you start, you need to run the following command (in terminal or in notebook beginning with `!` ) to download the datasets:
<!-- #endregion -->

```sh
# This command will download required datasets and put it in "./datasets".
!python get_datasets.py
```

## Implementation

These notebooks have the implementation and visualizations:

1. `knn.ipynb`
2. `linear_regression.ipynb`
3. `logistic_regression.ipynb`
4. `neural_network.ipynb`
5. `classification_nn.ipynb`

These notebooks require the following implementations:

1. `algorithms/knn.py`: KNN algorithm
2. `algorithms/linear_regression.py`: linear regression algorithm
3. `algorithms/logistic_regression.py`: logistic regression algorithm
4. `layers/linear.py`: linear layers with arbitrary input and output dimensions
5. `layers/relu.py`: ReLU activation function, forward and backward pass
6. `layers/softmax.py`: softmax function to calculate class probabilities
7. `layers/loss_func.py`: CrossEntropy loss, forward and backward pass
8. `layers/sequential.py`: Sequential layer composition
9. `utils/trainer.py`: Model training utility
10. `utils/optimizer.py`: Model optimization utility

```sh

```
