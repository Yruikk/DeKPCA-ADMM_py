# DeKPCA-ADMM_py
The code is the implementation of a decentralized algorithm for Kernel Principal Component Analysis (DeKPCA) in a sample-distributed setting, as described in the paper 
["A decentralized framework for kernel PCA with projection consensus constraints"](https://arxiv.org/abs/2211.15953). In this setup, each local agent holds a subset of samples with complete features.

This is the Python code of our paper, which uses mpi4py to accomplish a truly parallel setting.

Each folder in "ExperimentsForFigAndTable" corresponds to the code for a specific figure or table in the article. We conducted experiments on TomsHardware and Twitter datasets and have uploaded the source data for TomsHardware (as the Twitter data is too large, readers are advised to download it themselves). Before running any code, please make sure to copy the source data to the current directory.
