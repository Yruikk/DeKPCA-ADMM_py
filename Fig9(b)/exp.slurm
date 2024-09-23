#!/bin/bash

#SBATCH --job-name=exp1.1Twitter
#SBATCH --partition=cpu
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --mail-type=end           # 作业结束时，邮件提醒
#SBATCH --mail-user=ruikai.yang@sjtu.edu.cn

ulimit -s unlimited
ulimit -l unlimited

module load miniconda3 gcc mpich

source activate hypy

mpirun -np 6  python3 exp1.1Twitter.py