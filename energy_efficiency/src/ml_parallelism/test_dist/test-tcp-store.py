import torch.distributed as dist
from datetime import timedelta
import os

addr = os.environ['MASTER_ADDR']
port = int(os.environ['MASTER_PORT'])
world_size = int(os.environ['WORLD_SIZE'])
print(addr, port)
if os.environ['RANK'] == '0':
    # Run on process 1 (server)
    print('creating server')
    server_store = dist.TCPStore('127.0.0.1', 1234, world_size=world_size, is_master=True, timeout=timedelta(seconds=20))
    server_store.set("first_key", "first_value")
else: # Run on process 2 (client)
    print('creating client')
    client_store = dist.TCPStore('127.0.0.1', 1234, world_size=world_size, is_master=False, timeout=timedelta(seconds=20))
    print(client_store.get("first_key"))
