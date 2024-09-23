# DeKPCA-ADMM_py
The code is the implementation of a decentralized algorithm for Kernel Principal Component Analysis (DeKPCA) in a sample-distributed setting, as described in the paper 
["`
  [A decentralized framework for kernel PCA with projection consensus constraints](https://arxiv.org/abs/2211.15953)
"`](https://arxiv.org/abs/2211.15953)

"A decentralized framework for kernel PCA with projection consensus constraints". In this setup, each local agent holds a subset of samples with complete features.

This is the Python code of our paper, which uses mpi4py to accomplish a truly parallel setting.

Eigenvectors computed by SVD is regarded as ground truth.

Experiment1 shows effect of the number of neighbors.   
We compare results when the number of neighbors equals to 2,4,6,8,10 and 12.

Experiment2 shows effect of the number of local samples.  
We compare results when the number of local samples equals to 40,80,120,160,200 and 240.

Experiment3 shows effect of the number of network nodes.  
We compare results when the number of network nodes equals to 10,20,40,60 and 80.  
Based on the consideration of supercomputer computing efficiency, we did 5 experiments separately in Experiment3.
