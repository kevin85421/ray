import ray
import torch
import torch.distributed as dist
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.experimental.channel import ChannelContext
from verl import DataProto


def create_sample_dataproto(batch_size=8):
    """
    Creates a sample DataProto object with exactly the specified batch size
    """
    seq_length = 128

    # Common inputs for language models
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    position_ids = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, -1)

    # Create tensor dictionary
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Create non-tensor data
    raw_prompt_ids = [input_ids[i].tolist() for i in range(batch_size)]
    non_tensors = {"raw_prompt_ids": raw_prompt_ids}

    # Create DataProto using from_dict
    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    return data_proto


WORLD_SIZE = 2


@ray.remote
class Actor:
    def register_custom_serializer(self):
        TorchTensorType().register_custom_serializer()

    def setup(self, world_size, rank, init_method, group_name="default"):
        dist.init_process_group(
            backend="gloo", world_size=world_size, rank=rank, init_method=init_method
        )

    @ray.method(tensor_transport="nccl")
    def send_data_proto(self, batch_size):
        data_proto = create_sample_dataproto(batch_size)
        print("batch", type(data_proto.batch.get("attention_mask")))
        return data_proto

    def recv_data_proto(self, data_proto):
        in_actor_object_store = ray._private.worker.global_worker.in_actor_object_store
        print(f"in_actor_object_store: {in_actor_object_store}")
        print("recv", data_proto)
        return data_proto


if __name__ == "__main__":
    actors = [Actor.remote() for _ in range(WORLD_SIZE)]

    # TODO: Replace with an API call that takes in a list of actors and
    # returns a handle to the group.
    init_method = "tcp://localhost:8889"
    ray.get(
        [
            actor.setup.remote(WORLD_SIZE, rank, init_method)
            for rank, actor in enumerate(actors)
        ]
    )
    actor_ids = [actor._ray_actor_id for actor in actors]

    TorchTensorType().register_custom_serializer()

    # TODO(swang): Wrap actors in a Communicator interface.
    ctx = ChannelContext.get_current()
    ctx.communicators[0] = actors
    print("Collective group setup done")

    ray.get([actor.register_custom_serializer.remote() for actor in actors])
    print("Register custom serializer done")

    shape = (1,)

    batch_size = 3
    print("Sending an object consisting of a tensor to the same actor")
    ref = actors[0].send_data_proto.remote(batch_size)
    ref = actors[0].recv_data_proto.remote(ref)
    # Error
    print(ray.get(ref))
