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
    # os.environ['RANK'] = f'{self.rank}'
    # os.environ['WORLD_SIZE'] = '2'
    print(f"{self.rank} {self.rank==0} {os.environ['MASTER_ADDR']} attempting init")
    dist.init_process_group(
        "nccl",
        # backend='cuda:nccl',
        # backend=dist.Backend.NCCL,
        # backend='cpu:gloo,cuda:nccl',
        # backend='nccl',
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

    if self.rank == 0:
        # input = torch.tensor([[1,2],[3,4]], dtype=torch.float)
        x = torch.zeros(1)
        x = x.to(torch.device("cuda:0"))
        dist.send(tensor=x, dst=1)
    else:
        # output = torch.zeros([2,2], dtype=torch.float)
        y = torch.zeros(1)
        y = y.to(torch.device("cuda:1"))
        dist.recv(tensor=y, src=0)
        print(f"received {y}")

if __name__ == "__main__":
    rank = os.environ['RANK']
    local_rank = os.environ['LOCAL_RANK']
    pm0 = GroupMember(int(rank), torch.device(f"cuda:{local_rank}"))
    pm0.attempt()

# p0 = Process(target=pm0.attempt)

# p0.start()
# p0.join(timeout=25)
