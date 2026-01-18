import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel


# load pipeline
model_id = "stabilityai/sdxl-turbo"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir='./cache',
    use_safetensors=True).to("cuda")

# load finetuned model
unet_id = "mhdang/dpo-sdxl-text2image-v1"
unet = UNet2DConditionModel.from_pretrained(
    unet_id,
    subfolder="unet",
    torch_dtype=torch.float16)

pipe.unet = unet
pipe = pipe.to("cuda")

prompt = "Two cats playing chess on a tree branch"
image = pipe(prompt, guidance_scale=5).images[0].resize((512,512))

image.save("/content/drive/My Drive/generated_images/cats_playing_chess.png")