from datetime import timedelta
from multiprocessing import Process
import os
from threading import Thread
import torch
import torch.distributed as dist

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

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
    print(self.device)
    # os.environ['RANK'] = f'{self.rank}'
    # os.environ['WORLD_SIZE'] = '2'
    print(f"{self.rank} {self.rank==0} {os.environ['MASTER_ADDR']} attempting init")
    dist.init_process_group(
        # backend='cuda:nccl',
        # backend=dist.Backend.NCCL,
        backend='nccl',
        # backend='cpu:gloo',
        # init_method='env://',
        # store=dist.TCPStore(host_name=os.environ['MASTER_ADDR'], port=1234, is_master=self.rank==0, timeout=timedelta(seconds=20)),
        rank=self.rank,
        world_size=2,
        # device_id=self.device,
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
    input = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
    input = input * (self.rank + 1)
    input = input.to(self.device)
    self.print_barrier(self.rank, input)
    output = torch.empty([2,4], dtype=torch.int64)
    output = output.to(self.device)
    dist.all_to_all_single(output, input)
    print(f'{self.rank} successfully sent')
    # self.out = output
    self.print_barrier(self.rank, output)

rank = os.environ['RANK']
local_rank = os.environ['LOCAL_RANK']
pm0 = GroupMember(int(rank), torch.device(f"cuda:{local_rank}"))
p0 = Process(target=pm0.attempt)

p0.start()
p0.join(timeout=25)
