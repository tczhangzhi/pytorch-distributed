# Distribution is all you need

## Take-Away

笔者使用 PyTorch 编写了不同加速库在 ImageNet 上的使用示例（单机多卡），需要的同学可以当作 quickstart 将需要的部分 copy 到自己的项目中（Github 请点击下面链接）：

1. **[nn.DataParallel ](https://github.com/tczhangzhi/pytorch-distributed/blob/master/dataparallel.py) 简单方便的 nn.DataParallel**
2. **[torch.distributed](https://github.com/tczhangzhi/pytorch-distributed/blob/master/distributed.py) 使用 torch.distributed 加速并行训练**
3. **[torch.multiprocessing](https://github.com/tczhangzhi/pytorch-distributed/blob/master/multiprocessing_distributed.py) 使用 torch.multiprocessing 取代启动器**
4. **[apex](https://github.com/tczhangzhi/pytorch-distributed/blob/master/apex_distributed.py) 使用 apex 再加速**
5. **[horovod](https://github.com/tczhangzhi/pytorch-distributed/blob/master/horovod_distributed.py)** **horovod 的优雅实现**
6. **[slurm](https://github.com/tczhangzhi/pytorch-distributed/blob/master/distributed_slurm_main.py) GPU 集群上的分布式**
7. **补充：分布式 [evaluation](https://github.com/tczhangzhi/pytorch-distributed/blob/master/distributed.py)**

这里，笔者记录了使用 4 块 Tesla V100-PICE 在 ImageNet 进行了运行时间的测试，测试结果发现 **Apex 的加速效果最好，但与 Horovod/Distributed 差别不大**，平时可以直接使用内置的 Distributed。**Dataparallel 较慢，不推荐使用**。（后续会补上 V100/K80 上的测试结果，穿插了一些试验所以中断了）

![img](https://pic3.zhimg.com/80/v2-9c129a594c73aa8f2085dd17701de0e2_hd.jpg)

简要记录一下不同库的分布式训练方式：

## 简单方便的 nn.DataParallel

> DataParallel 可以帮助我们（使用单进程控）将模型和数据加载到多个 GPU 中，控制数据在 GPU 之间的流动，协同不同 GPU 上的模型进行并行训练（细粒度的方法有 scatter，gather 等等）。

DataParallel 使用起来非常方便，我们只需要用 DataParallel 包装模型，再设置一些参数即可。需要定义的参数包括：参与训练的 GPU 有哪些，device_ids=gpus；用于汇总梯度的 GPU 是哪个，output_device=gpus[0] 。DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总：

```
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
```

值得注意的是，模型和数据都需要先 load 进 GPU 中，DataParallel 的 module 才能对其进行处理，否则会报错：

```
# 这里要 model.cuda()
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      # 这里要 images/target.cuda()
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

汇总一下，DataParallel 并行训练部分主要与如下代码段有关：

```
# main.py
import torch
import torch.distributed as dist

gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_dataset = ...

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=...)

model = ...
model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

在使用时，使用 python 执行即可：

```
python main.py
```

在 ImageNet 上的完整训练代码，请点击[Github](https://link.zhihu.com/?target=https%3A//github.com/tczhangzhi/pytorch-distributed/blob/master/dataparallel.py)。

## 使用 torch.distributed 加速并行训练

> 在 pytorch 1.0 之后，官方终于对分布式的常用方法进行了封装，支持 all-reduce，broadcast，send 和 receive 等等。通过 MPI 实现 CPU 通信，通过 NCCL 实现 GPU 通信。官方也曾经提到用 DistributedDataParallel 解决 DataParallel 速度慢，GPU 负载不均衡的问题，目前已经很成熟了～

与 DataParallel 的单进程控制多 GPU 不同，在 distributed 的帮助下，我们只需要编写一份代码，torch 就会自动将其分配给 ![[公式]](https://www.zhihu.com/equation?tex=n) 个进程，分别在 ![[公式]](https://www.zhihu.com/equation?tex=n) 个 GPU 上运行。

在 API 层面，pytorch 为我们提供了 torch.distributed.launch 启动器，用于在命令行分布式地执行 python 文件。在执行过程中，启动器会将当前进程的（其实就是 GPU的）index 通过参数传递给 python，我们可以这样获得当前进程的 index：

```
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)
```

接着，使用 init_process_group 设置GPU 之间通信使用的后端和端口：

```
dist.init_process_group(backend='nccl')
```

之后，使用 DistributedSampler 对数据集进行划分。如此前我们介绍的那样，它能帮助我们将每个 batch 划分成几个 partition，在当前进程中只需要获取和 rank 对应的那个 partition 进行训练：

```
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

然后，使用 DistributedDataParallel 包装模型，它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）。all reduce 后不同 GPU 中模型的梯度均为 all reduce 之前各 GPU 梯度的均值：

```
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
```

最后，把数据和模型加载到当前进程使用的 GPU 中，正常进行正反向传播：

```
torch.cuda.set_device(args.local_rank)

model.cuda()

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

汇总一下，torch.distributed 并行训练部分主要与如下代码段有关：

```
# main.py
import torch
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

在使用时，调用 torch.distributed.launch 启动器启动：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

在 ImageNet 上的完整训练代码，请点击[Github](https://link.zhihu.com/?target=https%3A//github.com/tczhangzhi/pytorch-distributed/blob/master/distributed.py)。

## 使用 torch.multiprocessing 取代启动器

> 有的同学可能比较熟悉 torch.multiprocessing，也可以手动使用 torch.multiprocessing 进行多进程控制。绕开 torch.distributed.launch 自动控制开启和退出进程的一些小毛病～

使用时，只需要调用 torch.multiprocessing.spawn，torch.multiprocessing 就会帮助我们自动创建进程。如下面的代码所示，spawn 开启了 nprocs=4 个进程，每个进程执行 main_worker 并向其中传入 local_rank（当前进程 index）和 args（即 4 和 myargs）作为参数：

```
import torch.multiprocessing as mp

mp.spawn(main_worker, nprocs=4, args=(4, myargs))
```

这里，我们直接将原本需要 torch.distributed.launch 管理的执行内容，封装进 main_worker 函数中，其中 proc 对应 local_rank（当前进程 index），进程数 nproc 对应 4， args 对应 myargs：

```
def main_worker(proc, nproc, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
   torch.cuda.set_device(args.local_rank)

   train_dataset = ...
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

   model = ...
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

   optimizer = optim.SGD(model.parameters())

   for epoch in range(100):
      for batch_idx, (data, target) in enumerate(train_loader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

在上面的代码中值得注意的是，由于没有 torch.distributed.launch 读取的默认环境变量作为配置，我们需要手动为 init_process_group 指定参数：

```
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
```

汇总一下，添加 multiprocessing 后并行训练部分主要与如下代码段有关：

```
# main.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

mp.spawn(main_worker, nprocs=4, args=(4, myargs))

def main_worker(proc, nprocs, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
   torch.cuda.set_device(args.local_rank)

   train_dataset = ...
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

   model = ...
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

   optimizer = optim.SGD(model.parameters())

   for epoch in range(100):
      for batch_idx, (data, target) in enumerate(train_loader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

在使用时，直接使用 python 运行就可以了：

```
python main.py
```

在 ImageNet 上的完整训练代码，请点击[Github](https://link.zhihu.com/?target=https%3A//github.com/tczhangzhi/pytorch-distributed/blob/master/multiprocessing_distributed.py)。

## 使用 Apex 再加速

> Apex 是 NVIDIA 开源的用于混合精度训练和分布式训练库。Apex 对混合精度训练的过程进行了封装，改两三行配置就可以进行混合精度的训练，从而大幅度降低显存占用，节约运算时间。此外，Apex 也提供了对分布式训练的封装，针对 NVIDIA 的 NCCL 通信库进行了优化。

在混合精度训练上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。

```
from apex import amp

model, optimizer = amp.initialize(model, optimizer)
```

在分布式训练的封装上，Apex 在胶水层的改动并不大，主要是优化了 NCCL 的通信。因此，大部分代码仍与 torch.distributed 保持一致。使用的时候只需要将 torch.nn.parallel.DistributedDataParallel 替换为 apex.parallel.DistributedDataParallel 用于包装模型。在 API 层面，相对于 torch.distributed ，它可以自动管理一些参数（可以少传一点）：

```
from apex.parallel import DistributedDataParallel

model = DistributedDataParallel(model)
# # torch.distributed
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
```

在正向传播计算 loss 时，Apex 需要使用 amp.scale_loss 包装，用于根据 loss 值自动对精度进行缩放：

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
   scaled_loss.backward()
```

汇总一下，Apex 的并行训练部分主要与如下代码段有关：

```
# main.py
import torch
import argparse
import torch.distributed as dist

from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model, optimizer = amp.initialize(model, optimizer)
model = DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
         scaled_loss.backward()
      optimizer.step()
```

在使用时，调用 torch.distributed.launch 启动器启动：

```
UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

在 ImageNet 上的完整训练代码，请点击[Github](https://link.zhihu.com/?target=https%3A//github.com/tczhangzhi/pytorch-distributed/blob/master/apex_distributed.py)。

## Horovod 的优雅实现

> Horovod 是 Uber 开源的深度学习工具，它的发展吸取了 Facebook "Training ImageNet In 1 Hour" 与百度 "Ring Allreduce" 的优点，可以无痛与 PyTorch/Tensorflow 等深度学习框架结合，实现并行训练。

在 API 层面，Horovod 和 torch.distributed 十分相似。在 mpirun 的基础上，Horovod 提供了自己封装的 horovodrun 作为启动器。

与 torch.distributed.launch 相似，我们只需要编写一份代码，horovodrun 启动器就会自动将其分配给 ![[公式]](https://www.zhihu.com/equation?tex=n) 个进程，分别在 ![[公式]](https://www.zhihu.com/equation?tex=n) 个 GPU 上运行。在执行过程中，启动器会将当前进程的（其实就是 GPU的）index 注入 hvd，我们可以这样获得当前进程的 index：

```
import horovod.torch as hvd

hvd.local_rank()
```

与 init_process_group 相似，Horovod 使用 init 设置GPU 之间通信使用的后端和端口:

```
hvd.init()
```

接着，使用 DistributedSampler 对数据集进行划分。如此前我们介绍的那样，它能帮助我们将每个 batch 划分成几个 partition，在当前进程中只需要获取和 rank 对应的那个 partition 进行训练：

```
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

之后，使用 broadcast_parameters 包装模型参数，将模型参数从编号为 root_rank 的 GPU 复制到所有其他 GPU 中：

```
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
```

然后，使用 DistributedOptimizer 包装优化器。它能帮助我们为不同 GPU 上求得的梯度进行 all reduce（即汇总不同 GPU 计算所得的梯度，并同步计算结果）。all reduce 后不同 GPU 中模型的梯度均为 all reduce 之前各 GPU 梯度的均值：

```
hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16)
```

最后，把数据加载到当前 GPU 中。在编写代码时，我们只需要关注正常进行正向传播和反向传播：

```
torch.cuda.set_device(args.local_rank)

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

汇总一下，Horovod 的并行训练部分主要与如下代码段有关：

```
# main.py
import torch
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model.cuda()

optimizer = optim.SGD(model.parameters())

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
       images = images.cuda(non_blocking=True)
       target = target.cuda(non_blocking=True)
       ...
       output = model(images)
       loss = criterion(output, target)
       ...
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
```

在使用时，调用 horovodrun 启动器启动：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 --verbose python main.py
```

在 ImageNet 上的完整训练代码，请点击[Github](https://link.zhihu.com/?target=https%3A//github.com/tczhangzhi/pytorch-distributed/blob/master/horovod_distributed.py)。

## GPU 集群上的分布式

> Slurm，是一个用于 Linux 系统的免费、开源的任务调度工具。它提供了三个关键功能。第一，为用户分配资源(计算机节点)，以供用户执行工作。第二，它提供了一个框架，用于执行在节点上运行着的任务(通常是并行的任务)，第三，为任务队列合理地分配资源。如果你还没有部署 Slurm 可以按照笔者总结的[部署教程](https://zhuanlan.zhihu.com/p/149771261)进行部署。

通过运行 slurm 的控制命令，slurm 会将写好的 python 程序在每个节点上分别执行，调用节点上定义的 GPU 资源进行运算。要编写能被 Slurm 在 GPU 集群上执行的 python 分布式训练程序，我们只需要对上文中多进程的 DistributedDataParallel 代码进行修改，告诉每一个执行的任务（每个节点上的 python 程序），要用哪些训练哪一部分数据，反向传播的结果如何合并就可以了。

我们首先需要获得每个任务（对应每个节点）的基本信息，以便针对任务的基本信息处理其应当负责的数据。在使用 slurm 执行 srun python 代码时，python 可以从环境变量 os.environ 中获取当前 python 进程的基本信息：

```
import os
local_rank = os.environ['SLURM_PROCID'] # 当前任务的编号（比如节点 1 执行 1 号任务，节点 2 执行 2 号任务）
world_size = os.environ['SLURM_NPROCS'] # 共开启的任务的总数（共有 2 个节点执行了 2 个任务）
job_id = os.environ['SLURM_JOBID'] # 当前作业的编号（这是第 1 次执行 srun，编号为 1）
```

在每个任务（节点）中，我们需要为节点中的每个 GPU 资源分配一个进程，管理该 GPU 应当处理的数据。

当前节点的 GPU 的数量可以由 torch.cuda 查询得到：

```
ngpus_per_node = torch.cuda.device_count()
```

接着，与上文相似，我们使用 torch.multiprocessing 创建 ngpus_per_node 个进程，其中，每个进程执行的函数为 main_worker ，该函数调用所需要的由 args 传入：

```
mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
```

在编写 main_worker 时，我们首先需要解决的问题是：不同节点、或者同一节点间的不同进程之间需要通信来实现数据的分割、参数的合并。我们可以使用 pytorch 的 dist 库在共享文件系统上创建一个文件进行通信：

```
import torch.distributed as dist

def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
```

完成进程创建和通信后，下一步就是实现我们常用的 pipline 了，即加载模型、加载数据、正向传播、反向传播。与上文相似，这里，我们把模型加载进当前进程所对应的 GPU 中：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
```

接着，把当前进程对应的数据段采样出来，也加载到对应的 GPU 中。同样可以使用 pytorch 的 dist 库实现这个采样过程：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
  ...
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=train_sampler)
  for i, (images, target) in enumerate(train_loader):
    images = images.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)
```

最后，进行正常的正向和反向传播：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
  ...
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=train_sampler)
  for i, (images, target) in enumerate(train_loader):
    images = images.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)
    ...
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

在使用时，调用 srun 启动任务：

```
srun -N2 --gres gpu:1 python distributed_slurm_main.py --dist-file dist_file
```

在 ImageNet 上的完整训练代码，请点击[Github](https://github.com/tczhangzhi/pytorch-distributed/blob/master/distributed_slurm_main.py)。

## 分布式 evaluation

> all_reduce, barrier 等 API 是 distributed 中更为基础和底层的 API。这些 API 可以帮助我们控制进程之间的交互，控制 GPU 数据的传输。在自定义 GPU 协作逻辑，汇总 GPU 间少量的统计信息时，大有用处。熟练掌握这些 API 也可以帮助我们自己设计、优化分布式训练、测试流程。

到目前为止，Distributed Sampler 能够帮助我们分发数据，DistributedDataParallel、hvd.broadcast_parameters 能够帮助我们分发模型，并在框架的支持下解决梯度汇总和参数更新的问题。然而，还有一些同学还有这样的疑惑，

1. 训练样本被切分成了若干个部分，被若干个进程分别控制运行在若干个 GPU 上，如何在进程间进行通信汇总这些（GPU 上的）信息？
2. 使用一张卡进行推理、测试太慢了，如何使用 Distributed 进行分布式地推理和测试，并将结果汇总在一起？
3. ......

要解决这些问题，我们缺少一个更为基础的 API，**汇总记录不同 GPU 上生成的准确率、损失函数等指标信息**。这个 API 就是 `torch.distributed.all_reduce`。示意图如下：

![img](https://pic4.zhimg.com/80/v2-f424bdc8108abd5421e3af3b902b2ccf_720w.jpg)

具体来说，它的工作过程包含以下三步：

1. 通过调用 `all_reduce(tensor, op=...)`，当前进程会向其他进程发送 `tensor`（例如 rank 0 会发送 rank 0 的 tensor 到 rank 1、2、3）
2. 接受其他进程发来的 `tensor`（例如 rank 0 会接收 rank 1 的 tensor、rank 2 的 tensor、rank 3 的 tensor）。
3. 在全部接收完成后，当前进程（例如rank 0）会对当前进程的和接收到的 `tensor` （例如 rank 0 的 tensor、rank 1 的 tensor、rank 2 的 tensor、rank 3 的 tensor）进行 `op` （例如求和）操作。

使用 `torch.distributed.all_reduce(loss, op=torch.distributed.reduce_op.SUM)`，我们就能够对不数据切片（不同 GPU 上的训练数据）的损失函数进行求和了。接着，我们只要再将其除以进程（GPU）数量 `world_size`就可以得到损失函数的平均值。

正确率也能够通过同样方法进行计算：

```
# 原始代码
output = model(images)
loss = criterion(output, target)
        
acc1, acc5 = accuracy(output, target, topk=(1, 5))
losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
​
# 修改后，同步各 GPU 中数据切片的统计信息，用于分布式的 evaluation
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
​
output = model(images)
loss = criterion(output, target)
acc1, acc5 = accuracy(output, target, topk=(1, 5))
​
torch.distributed.barrier()
​
reduced_loss = reduce_tensor(loss.data)
reduced_acc1 = reduce_tensor(acc1)
reduced_acc5 = reduce_tensor(acc5)
​
losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
```

值得注意的是，为了同步各进程的计算进度，我们在 reduce 之前插入了一个同步 API `torch.distributed.barrier()`。在所有进程运行到这一步之前，先完成此前代码的进程会等待其他进程。这使得我们能够得到准确、有序的输出。在 Horovod 中，我们无法使用 `torch.distributed.barrier()`，取而代之的是，我们可以在 allreduce 过程中指明：

```
def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    rt /= world_size
    return rt
    
output = model(images)
loss = criterion(output, target)
acc1, acc5 = accuracy(output, target, topk=(1, 5))

reduced_loss = reduce_tensor(loss.data)
reduced_acc1 = reduce_tensor(acc1)
reduced_acc5 = reduce_tensor(acc5)

losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
```