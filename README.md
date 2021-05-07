# DKPCA-ADMM
A Decentralized algorithm for Kernel Principal Component Analysis (DeKPCA) for sample-distributed setting, where each local agent contains a subset of samples with full features.

This is Python code of our paper, which uses mpi4py to accomplish truly parallel setting.

Eigenvectors computed by SVD is regarded as ground truth.

Experiment1 shows effect of the number of neighbors. 
We compare results when the number of neighbors equals to 2,4,6,8,10 and 12.

Experiment2 shows effect of the number of local samples. 
We compare results when the number of local samples equals to 40,80,120,160,200,240 and 180.

Experiment3 shows effect of the number of network nodes.
We compare results when the number of network nodes equals to 10,20,40,60 and 80.
Because we can only test one J(the number of network nodes) value in an experiment, we did 5 experiments separately in Experiment3.
