import Pyro5.api
import torch
import numpy as np
from typing import Optional
import time

from cosmos1.models.diffusion.inference.inference_utils import load_model_by_config, load_network_model
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.models.diffusion.conditioner import BaseVideoCondition, VideoExtendCondition

import msgpack
import msgpack_numpy as m

Pyro5.config.SERIALIZER = "msgpack"
Pyro5.config.SERVERTYPE = "multiplex"
# Ensure msgpack_numpy is patched
m.patch()

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class RemoteDenoiser:
    def __init__(self):            
        self.model_name = "Cosmos_1_0_Diffusion_Text2World_7B"
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionT2WModel,
        )        
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_name = "Cosmos-1.0-Diffusion-7B-Text2World"
        load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}/model.pt")
        print(f"DiffusionModel: precision {self.model.precision}")
        
                
    @torch.no_grad()
    def remote_denoise(self,                     
                noise_x_bytes: bytes,
                sigma_bytes: bytes,
                condition_type: str,
                condition_bytes: bytes
                ) -> bytes:                
        try:                
            start_time = time.time()  # Start timing
            noise_x = torch.tensor(m.unpackb(noise_x_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])
            sigma = torch.tensor(m.unpackb(sigma_bytes), device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"])
            
            if condition_type == "BaseVideoCondition":
                condition_dict = m.unpackb(condition_bytes)
                # Convert numpy arrays to tensors in condition_dict
                condition_dict = {
                    key: torch.tensor(val, device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"]) 
                    if isinstance(val, np.ndarray) else val
                    for key, val in condition_dict.items()
                }
                condition = BaseVideoCondition(**condition_dict)
            elif condition_type == "VideoExtendCondition":
                condition_dict = m.unpackb(condition_bytes)
                # Convert numpy arrays to tensors in condition_dict
                condition_dict = {
                    key: torch.tensor(val, device=self.model.tensor_kwargs["device"], dtype=self.model.tensor_kwargs["dtype"]) 
                    if isinstance(val, np.ndarray) else val
                    for key, val in condition_dict.items()
                }
                condition = VideoExtendCondition(**condition_dict)
            else:
                raise ValueError(f"Unknown condition type: {condition_type}")
                                
            x0 = self.model.denoise(noise_x, sigma, condition=condition).x0                        
            x0_bytes = x0.float().cpu().numpy()
            end_time = time.time()  # End timing
            print(f"Denoising took {end_time - start_time:.2f} seconds")
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