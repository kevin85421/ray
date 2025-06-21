import ray
from ray.experimental.collective import (
    create_collective_group,
    destroy_collective_group,
)
import torch


@ray.remote
class VLLMWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id

    @ray.method(tensor_transport="gloo")
    def produce(self, data):
        return data


@ray.remote
class FSDPWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def consume(self, data):
        return data


NUM_VLLM_WORKERS = 4
NUM_FSDP_WORKERS = 4

vllm_group_1 = [VLLMWorker.remote(i) for i in range(NUM_VLLM_WORKERS)]
fsdp_group_1 = [FSDPWorker.remote(i) for i in range(NUM_FSDP_WORKERS)]

group = create_collective_group(vllm_group_1 + fsdp_group_1, backend="torch_gloo")


fsdp_refs = []
for i in range(len(vllm_group_1)):
    tensor = torch.tensor([i])
    ref = vllm_group_1[i].produce.remote(tensor)
    fsdp_refs.append(fsdp_group_1[i].consume.remote(ref))

tensors = ray.get(fsdp_refs)
assert len(tensors) == NUM_FSDP_WORKERS
assert all(torch.equal(tensor, torch.tensor([i])) for i, tensor in enumerate(tensors))

# Simulate the scaling up of the vLLM group.
vllm_group_2 = [VLLMWorker.remote(i) for i in range(NUM_VLLM_WORKERS)]

# Destroy the old group and create a new one to include the new vLLM group.
destroy_collective_group(group)
create_collective_group(
    vllm_group_1 + vllm_group_2 + fsdp_group_1, backend="torch_gloo"
)

fsdp_refs = []
for i in range(len(vllm_group_2)):
    tensor = torch.tensor([i])
    ref = vllm_group_2[i].produce.remote(tensor)
    fsdp_refs.append(fsdp_group_1[i].consume.remote(ref))

tensors = ray.get(fsdp_refs)
assert len(tensors) == NUM_FSDP_WORKERS
assert all(torch.equal(tensor, torch.tensor([i])) for i, tensor in enumerate(tensors))
