#!/bin/bash

#SBATCH --job-name=exp_Table_20
#SBATCH --partition=cpu
#SBATCH -n 20
#SBATCH --ntasks-per-node=20
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --mail-type=end           # 作业结束时，邮件提醒
#SBATCH --mail-user=ruikai.yang@sjtu.edu.cn

module load miniconda3 gcc mpich intel-mpi

source activate hypy

mpiexec -n 20 python exp_Table2.py