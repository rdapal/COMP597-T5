from datetime import timedelta
from multiprocessing import Process
import os
from threading import Thread
import torch
import torch.distributed as dist
from typing import List

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

class GroupMember():
  def __init__(self, rank : int, device : torch.device, input : List[torch.Tensor]):
    self.rank = rank
    self.device = device
    self.input = input

  def print_barrier(self, *args):
    dist.barrier()
    if self.rank == 0:
      print(*args)
    dist.barrier()
    if self.rank == 1:
      print(*args)
  
  def attempt(self):
    print(f'{self.rank} attempting init')
    dist.init_process_group(
        backend='cpu:gloo,cuda:nccl',
        rank=self.rank,
        world_size=2,
        device_id=self.device,
        # device_id=None,
        timeout=timedelta(seconds=20)
    )
    # print(f'{self.rank} attempting group creation')
    # group = dist.new_group(
    #     ranks=[0,1],
    #     backend='nccl',
    #     device_id=device,
    #     timeout=timedelta(seconds=20)
    # )
    print(f'{self.rank} attempting send')
    self.print_barrier(self.rank, self.input)
    output = [torch.zeros([2], dtype=torch.int64), torch.zeros([2,2], dtype=torch.int64)]
    output = [t.to(self.device) for t in output]
    dist.all_to_all(output, self.input)
    print(f'{self.rank} successfully sent')
    # self.out = output
    self.print_barrier(self.rank, output)

input0 = [torch.tensor([1,2]), torch.tensor([3,4])]
# input1 = [torch.tensor([5,6]), torch.tensor([[10,11],[12,13]])]

rank = int(os.environ['RANK'])
inputs = [torch.tensor([1,2]), torch.tensor([3,4])]
if rank != 0:
    inputs = [torch.tensor([5,6]), torch.tensor([[10,11],[12,13]])]
    # inputs = [torch.tensor([5,6]), torch.tensor([7,8])]
device = torch.device(f"cuda:{rank}")
inputs = [t.to(device) for t in inputs]
pm0 = GroupMember(rank, torch.device(f"cuda:{rank}"), inputs)
# pm1 = GroupMember(1, torch.device("cuda:1"), input1)
# p0 = Process(target=pm0.attempt)
# p1 = Process(target=pm1.attempt)

pm0.attempt()
dist.destroy_process_group()
