from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

import sys

model_type = sys.argv[1]
if model_type == 'hd':
    ootd_model = OOTDiffusionHD(1)
elif model_type == 'dc':
    ootd_model = OOTDiffusionDC(1)

def create_unetgarm_input():
    pass

def myunet_garm(garm_latents, prompt_embeds):
    _, spatial_attn_outputs = ootd_model.unet_garm(
            garm_latents, # [2, 4, 128, 96]
            0,
            encoder_hidden_states=prompt_embeds, # [2, 2, 768]
            return_dict=False)
    return spatial_attn_outputs

def create_unetvton_input():
    pass

def myunet_vton(latent_vton_model_input, spatial_attn_inputs, t, prompt_embeds):
    noise_pred = ootd_model.unet_vton(latent_vton_model_input,
                         spatial_attn_inputs,
                         t,
                         prompt_embeds,
                         return_dict=False)
    return noise_pred[0]

