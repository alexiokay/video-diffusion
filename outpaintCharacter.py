
# 1. Install the required packages:
# !pip install git+https://github.com/lucidrains/stable-soft-diffusion.git
# !pip install torch torchvision matplotli

# 2. Import the required modules:
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from soft_diffusion import models, utils


# 3. Load the model and set the device:
model = models.StableDiffusionGenerator(
    image_size=256,
    image_channels=3,
    num_res_blocks=2,
    diffusion_steps=1000,
    dropout=0.1,
    use_new_attention_order=True,
    use_fp16=False,
    n_group=16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 4. Load the image and preprocess it:
img_path = "path/to/your/image.jpg"
img = Image.open(img_path)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_tensor = preprocess(img).unsqueeze(0).to(device)

# 5. Generate the outpainted image:
noise = torch.randn(1, 3, 256, 256).to(device)
outpaint, _ = model.sample_conditioned_on_image(img_tensor, noise, diffusion_steps=1000, eps=1e-5, progress=True)
outpaint = utils.normalize_image(outpaint.squeeze(0).cpu().detach().numpy())

# 6. Save and show the outpainted image:
plt.imshow(outpaint.transpose(1, 2, 0))
plt.show()
