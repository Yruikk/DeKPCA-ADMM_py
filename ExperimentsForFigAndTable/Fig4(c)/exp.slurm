#!/bin/bash

#SBATCH --job-name=expA2.2Tom
#SBATCH --partition=cpu
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --mail-type=end           # 作业结束时，邮件提醒
#SBATCH --mail-user=ruikai.yang@sjtu.edu.cn

module load miniconda3 gcc mpich intel-mpi

source activate hypy

mpiexec -n 10 python expA2.2Tom.py