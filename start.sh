CUDA_VISIBLE_DEVICES=0,1,2,3 python multiprocessing_distributed.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 apex_distributed.py
HOROVOD_WITH_PYTORCH=1 CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 --verbose python horovod_distributed.py
srun -N2 --gres gpu:4 python distributed_slurm_main.py --dist-file dist_file