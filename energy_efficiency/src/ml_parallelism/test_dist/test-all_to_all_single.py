from datetime import timedelta
from multiprocessing import Process
import os
from threading import Thread
import torch
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

class GroupMember():
  def __init__(self, rank : int, device : torch.device):
    self.rank = rank
    self.device = device

  def print_barrier(self, *args):
    dist.barrier()
    if self.rank == 0:
      print(*args)
    dist.barrier()
    if self.rank == 1:
      print(*args)
  
  def attempt(self):
    os.environ['RANK'] = f'{self.rank}'
    os.environ['WORLD_SIZE'] = '2'
    print(f'{self.rank} attempting init')
    dist.init_process_group(
        # backend='cuda:nccl',
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
    input = torch.tensor([[1,2],[3,4]])
    input = input * (self.rank + 1)
    self.print_barrier(self.rank, input)
    output = torch.empty([4], dtype=torch.int64)
    dist.all_to_all_single(output, input)
    print(f'{self.rank} successfully sent')
    # self.out = output
    self.print_barrier(self.rank, output)

pm0 = GroupMember(0, torch.device("cuda:0"))
pm1 = GroupMember(1, torch.device("cuda:1"))
p0 = Process(target=pm0.attempt)
p1 = Process(target=pm1.attempt)

p0.start()
p1.start()
p0.join(timeout=25)
p1.join(timeout=25)
