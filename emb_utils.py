from torch.nn.functional import silu
import torch

def get_time_emb(self,t_steps,return_after=False):

    time_emb_list=[]
    for sigma in t_steps:

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_noise = sigma.log() / 4
        # print(f"c_noise={c_noise.item()}")

        time_emb = self.model.get_time_emb_model(c_noise.flatten(),return_after)
        time_emb_list.append(time_emb)
    return time_emb_list
    
def get_time_emb_model(self,noise_labels,return_after):
    emb = self.map_noise(noise_labels)
    emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
    if return_after == False:
        # print('return_after=False')
        return emb
    
    emb = silu(self.map_layer0(emb))
    emb = silu(self.map_layer1(emb))

    return emb

