import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, UNet2DConditionModel


def get_model(model):
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            cache_dir='./cache'
            )
    elif model == 'sd21':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            cache_dir='./cache'
            )
    elif model == 'sd21-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='./cache'
            )
    elif model == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir='./cache'
            )
    elif model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='./cache'
            )
    elif model == 'fine-tune':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True)

        # load finetuned model
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1",
            subfolder="unet",
            torch_dtype=torch.float16)

        pipe.unet = unet

    return pipe