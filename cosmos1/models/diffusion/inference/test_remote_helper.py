import Pyro5.api
import torch
from cosmos1.models.diffusion.model.model_t2w import CosmosCondition
import numpy as np
import msgpack
import msgpack_numpy as m

Pyro5.config.SERIALIZER = "msgpack"
# Ensure msgpack_numpy is patched
m.patch()


with Pyro5.api.Proxy("PYRO:remote_denoiser@0.0.0.0:9090") as proxy:    
    with Pyro5.api.Proxy("PYRO:remote_denoiser@0.0.0.0:9090") as proxy:
        # Remote denoise call with None checks
        x0_bytes = proxy.remote_denoise(
            m.packb(torch.randn([1, 16, 16, 88, 160]).float().cpu().numpy()),
            m.packb(torch.randn([1]).float().cpu().numpy()),
            m.packb(torch.randn([1, 512, 1024]).float().cpu().numpy()),
            m.packb(torch.randn([1, 512]).float().cpu().numpy()),            
            m.packb(np.array([])),
            m.packb(np.array([]))
        )
    x0 = torch.tensor(m.unpackb(x0_bytes))
    print(f"x0: {x0}")

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