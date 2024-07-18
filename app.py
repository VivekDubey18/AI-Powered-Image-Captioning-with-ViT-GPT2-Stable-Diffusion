from diffusers import StableDiffusionPipeline
import gradio as gr
import torch

# Load the model
model_id1 =  "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

# Define a function to generate image from text prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Image(),
    title="Text-to-Image Generator",
    description="Enter a text prompt to generate an image.           Developed by VIVEK Dubey"
)

# Launch the interface
interface.launch()
