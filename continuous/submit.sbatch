#!/bin/bash
#SBATCH -J AAS   ### TASK NAME
#SBATCH --gres=gpu:a100:1  ### use gpu, a100 num is 1
#SBATCH -N 1            ### APPOINT USING ONE COMPUTE NODE
#SBATCH -n 1           ### appoint all task num 1
#SBATCH -o /home/chenzhenglong/Out_dir/AAS_test.out  ### appoint output file
#SBATCH -e /home/chenzhenglong/Out_dir/AAS_test.error  ###appoint error file
#SBATCH --qos=a100g1   ### apoint qos a100g1;nvlinkg8
#SBATCH -p a100     ### apoint queen a100;nvlink
### main continuous
### cd /home/chenzhenglong/Projects/Experiment/continuous
cd /home/chenzhenglong/RemoteProjects/AAS/code || exit
source activate pytorch
out_path=/home/chenzhenglong/RemoteProjects/AAS/out
srun -e $out_path/.error python general_k.py -o $out_path/.out
