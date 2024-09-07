import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import os

# Set environment variable to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VGG16 model
vgg = models.vgg16(pretrained=True).to(device)
vgg.eval()

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Hook class to capture intermediate outputs
class Hook:
    def __init__(self, module):
        self.input = None
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

# Function to get gradients for DeepDream
def get_gradients(net_in, net, layer):     
    net_in = net_in.unsqueeze(0).to(device)
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    loss = hook.output[0].norm()
    loss.backward()
    gradients = net_in.grad.detach().squeeze()
    hook.close()  # Ensure hook is removed after use
    return gradients / (torch.std(gradients) + 1e-8)  # Normalize gradients to enhance effects

# Denormalization image transform
denorm = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

# Function to run DeepDream with multiple octaves
def deepdream_octaves(image, iterations, lr, num_octaves=3, octave_scale=1.4):
    """ Runs DeepDream with multiple octaves to enhance the trippiness. """
    
    # Preprocess the image
    image_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image).to(device)
    
    octaves = [image_tensor]
    for _ in range(num_octaves - 1):
        new_size = [int(dim / octave_scale) for dim in octaves[-1].shape[1:]]
        octaves.append(torch.nn.functional.interpolate(octaves[-1].unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze(0))
    
    detail = torch.zeros_like(octaves[-1], device=device)
    
    for octave_idx in range(num_octaves):
        image_tensor = octaves[-1 - octave_idx] + detail
        
        for i in range(iterations):
            gradients = get_gradients(image_tensor, vgg, list(vgg.features.modules())[10])  # Adjust the layer index as needed
            gradients = gradients * (gradients > 0).float()  # Keep only positive gradients
            image_tensor = image_tensor + lr * gradients

        # Upscale and add details
        if octave_idx < num_octaves - 1:
            detail = image_tensor - octaves[-1 - octave_idx]
            detail = torch.nn.functional.interpolate(detail.unsqueeze(0), size=octaves[-2 - octave_idx].shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    # Denormalize and convert back to PIL image
    img_out = image_tensor.detach().cpu()
    img_out = denorm(img_out)
    img_out_np = img_out.numpy().transpose(1, 2, 0)
    img_out_np = np.clip(img_out_np, 0, 1)
    img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
    return img_out_pil
