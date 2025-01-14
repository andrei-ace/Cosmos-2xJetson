import Pyro5.api
import torch
import numpy as np
from typing import Optional

from cosmos1.models.diffusion.inference.inference_utils import load_model_by_config, load_network_model
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel, ndarray_to_list
from cosmos1.models.diffusion.conditioner import CosmosCondition

import msgpack
import msgpack_numpy as m

Pyro5.config.SERIALIZER = "msgpack"
# Ensure msgpack_numpy is patched
m.patch()

@Pyro5.api.expose
class RemoteDenoiser:
    def __init__(self):            
        self.model_name = "Cosmos_1_0_Diffusion_Text2World_7B"
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionT2WModel,
        )
        print(f"DiffusionModel: precision {self.model.precision}")
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_name = "Cosmos-1.0-Diffusion-7B-Text2World"
        load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}/model.pt")
                

    def remote_denoise(self,                     
                noise_x_bytes, sigma_bytes, 
                crossattn_emb_bytes, crossattn_mask_bytes, 
                padding_mask_data_bytes, 
                scalar_feature_data_bytes):
        try:
            with torch.no_grad():            
                noise_x = torch.tensor(m.unpackb(noise_x_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])
                sigma = torch.tensor(m.unpackb(sigma_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])
                
                # Create condition object with empty array checks
                padding_mask = None
                if padding_mask_data_bytes:
                    unpacked_padding = m.unpackb(padding_mask_data_bytes)
                    if isinstance(unpacked_padding, np.ndarray) and unpacked_padding.size > 0:
                        padding_mask = torch.tensor(unpacked_padding, device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])

                scalar_feature = None
                if scalar_feature_data_bytes:
                    unpacked_scalar = m.unpackb(scalar_feature_data_bytes)
                    if isinstance(unpacked_scalar, np.ndarray) and unpacked_scalar.size > 0:
                        scalar_feature = torch.tensor(unpacked_scalar, device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])

                condition = CosmosCondition(
                    crossattn_emb=torch.tensor(m.unpackb(crossattn_emb_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"]),
                    crossattn_mask=torch.tensor(m.unpackb(crossattn_mask_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"]),
                    # padding_mask=padding_mask,
                    # scalar_feature=scalar_feature
                )
                
                x0 = self.model.denoise(noise_x, sigma, condition=condition).x0
                print(f"x0: {x0.shape}")
        
            x0_bytes = x0.float().cpu().numpy()
            print(f"x0: {x0_bytes.shape}")
            return m.packb(x0_bytes)
        except Exception as e:
            import traceback
            print(f"Error in remote_denoise: {e}")
            print("Full stack trace:")
            print(traceback.format_exc())
            return None
    
def main():
    daemon = Pyro5.server.Daemon(host="0.0.0.0", port=9090)
    uri = daemon.register(RemoteDenoiser(), "remote_denoiser")
    print("Server started. URI:", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()