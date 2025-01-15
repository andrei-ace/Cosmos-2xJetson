import Pyro5.api
import torch
from cosmos1.models.diffusion.conditioner import BaseVideoCondition
from cosmos1.models.diffusion.model.model_t2w import CosmosCondition
import numpy as np
import msgpack
import msgpack_numpy as m
import time

Pyro5.config.SERIALIZER = "msgpack"
# Ensure msgpack_numpy is patched
m.patch()


with Pyro5.api.Proxy("PYRO:remote_denoiser@0.0.0.0:9090") as proxy:
    start_time = time.time()
    with Pyro5.api.Proxy("PYRO:remote_denoiser@0.0.0.0:9090") as proxy:
        # Remote denoise call with None checks
        condition_dict = BaseVideoCondition(
                crossattn_emb=torch.randn([1, 512, 1024]).float().cpu().numpy(),
                crossattn_mask=torch.randn([1, 512]).float().cpu().numpy(),
                padding_mask=torch.randn([1, 1, 704, 1280]).float().cpu().numpy(),
                fps=torch.randn([1]).float().cpu().numpy(),
                num_frames=torch.randn([1]).float().cpu().numpy(),
                image_size=torch.randn([1, 4]).float().cpu().numpy(),
                scalar_feature=None
            ).to_dict()
        condition_dict.pop('data_type', None)
        x0_bytes = proxy.remote_denoise(
            m.packb(torch.randn([1, 16, 16, 88, 160]).float().cpu().numpy()),
            m.packb(torch.randn([1]).float().cpu().numpy()),
            'BaseVideoCondition',
            m.packb(condition_dict),            
        )
        x0 = torch.tensor(m.unpackb(x0_bytes))
        end_time = time.time()
        print(f"x0: {x0.shape}")        
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

####
# noise_x: torch.Size([1, 16, 16, 88, 160])
# sigma: torch.Size([1])
# condition: BaseVideoCondition(crossattn_emb=tensor(,
#        device='cuda:0', dtype=torch.bfloat16), crossattn_mask=tensor(, device='cuda:0',
#        dtype=torch.bfloat16), data_type=<DataType.VIDEO: 'video'>, padding_mask=tensor(), device='cuda:0',
#        dtype=torch.bfloat16), fps=tensor([24.], device='cuda:0', dtype=torch.bfloat16), num_frames=tensor([121.], device='cuda:0', dtype=torch.bfloat16), image_size=tensor([[ 704., 1280.,  704., 1280.]], device='cuda:0', dtype=torch.bfloat16), scalar_feature=None)
# crossattn_emb: torch.Size([1, 512, 1024])
# crossattn_mask: torch.Size([1, 512])
# padding_mask: torch.Size([1, 1, 704, 1280])
# fps: torch.Size([1])
# num_frames: torch.Size([1])
# image_size: torch.Size([1, 4])