import torch
import PIL

from typing import List
from PIL import Image

from ip_adapter import IPAdapterXL
from ip_adapter.utils import get_generator

from .sdxl_sw_pipeline import SWStableDiffusionXLPipeline


class SWIPAdapterXL(IPAdapterXL):
    """SW guidance for IP Adapter SDXL"""

    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=["block"]):
        super().__init__(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens, target_blocks)
        self.image_encoder.to(self.device, dtype=torch.bfloat16)
        self.image_proj_model.to(dtype=torch.bfloat16)
        self.pipe.unet.to(dtype=torch.bfloat16)

    def generate(
        self,
        pil_image,
        sw_reference: PIL.Image = None,
        sw_steps: int = 8,
        sw_u_lr: float = 0.05 * 10**3,
        num_guided_steps: int = None,
        sw_debug: bool = False,
        #-----------------------------------
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        neg_content_emb=None,
        neg_content_prompt=None,
        neg_content_scale=1.0,
        **kwargs,
    ):
        if not isinstance(self.pipe, SWStableDiffusionXLPipeline):
            raise TypeError('Pipeline should be of type SWStableDiffusionXLPipeline')
            
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_, # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_, # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb
        else:
            pooled_prompt_embeds_ = None

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            sw_reference=sw_reference,
            sw_steps=sw_steps,
            sw_u_lr=sw_u_lr,
            num_guided_steps=num_guided_steps,
            sw_debug=sw_debug,
            #-----------------------------------
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images



