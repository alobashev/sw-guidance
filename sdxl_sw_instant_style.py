import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
from PIL import Image

from ip_adapter import IPAdapterXL

from sw_guidance import SWIPAdapterXL, SWStableDiffusionXLPipeline


if __name__ == "__main__":
    
    # ------------------------------------------    
    device = torch.device("cuda:1")
    base_model_path = "../../models/safetensors/RealVisXL_v4.safetensors"
    image_encoder_path = "sdxl_models/image_encoder"
    ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"

    # pipe = StableDiffusionXLPipeline.from_single_file(
    #     base_model_path,
    #     torch_dtype=torch.float16,
    #     add_watermarker=False,
    # )
    # # # reduce memory consumption
    # pipe.enable_vae_tiling()
    
    # # # load ip-adapter
    # # target_blocks=["block"] for original IP-Adapter
    # # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
    # # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
    # ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

    # image = "./assets/4.jpg"
    # image = Image.open(image)
    # image.resize((512, 512))
    
    # # generate image variations with only image prompt
    # images = ip_model.generate(pil_image=image,
    #                             prompt="a cat, masterpiece, best quality, high quality",
    #                             negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    #                             scale=1.0,
    #                             guidance_scale=5,
    #                             num_samples=1,
    #                             num_inference_steps=30, 
    #                             # seed=42,
    #                             #neg_content_prompt="a rabbit",
    #                             #neg_content_scale=0.5,
    #                           )
    
    num_inference_steps = 30
    num_guided_steps = 28  # or None
    guidance_scale = 5
    sw_u_lr = 0.01 * 10**4

    pipe = SWStableDiffusionXLPipeline.from_single_file(
        base_model_path,
        local_files_only=True,                                         
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # # load ip-adapter
    # target_blocks=["block"] for original IP-Adapter
    # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
    # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
    ip_model = SWIPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

    # image = "./sw_assets/van_gogh_bw.png"
    # image = "./sw_assets/origami_bw.png" 
    # image = "./sw_assets/van_gogh_2_square_bw.png"
    image = "origami_cat.jpg"
    image = Image.open(image)
    image.resize((512, 512))

    # sw_ref_im = "./sw_assets/origami.png"
    # sw_ref_im = "./sw_assets/pink_night_square.png"
    sw_ref_im = "../../data/misc/pastel.jpg"
    sw_ref_im = Image.open(sw_ref_im).convert('RGB')
    sw_ref_im = sw_ref_im.resize((512,512))

    # generate image variations with only image prompt
    images = ip_model.generate(pil_image=image,
                                prompt="a cat",
                                # negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch",
                                scale=1.5,
                                guidance_scale=5,
                                num_samples=1,
                                num_inference_steps=num_inference_steps,
                                # --------------------
                                num_guided_steps=num_guided_steps,  # or None
                                sw_u_lr=sw_u_lr,
                                height=768,
                                width=768,
                                sw_reference=sw_ref_im,
                                # seed=42,
                                #neg_content_prompt="a rabbit",
                                #neg_content_scale=0.5,
                              )


    
    images[0].save("result.png")


