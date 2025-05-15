# LDM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels // 4 if out_channels >=4 else 1), num_channels=out_channels)


    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class BasicDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels // 4 if out_channels >=4 else 1), num_channels=out_channels)

    def forward(self, x):
        return self.relu(self.norm(self.deconv(x)))

# VAE Autoencoder 
# responsible for encoding images into a lower-dimensional latent space
# and decoding latent representations back into images.
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims_encoder=None, hidden_dims_decoder=None, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.final_encoder_res = img_size // 8 # Assuming 3 downsampling steps of factor 2

        if hidden_dims_encoder is None:
            hidden_dims_encoder = [32, 64, 128]
        if hidden_dims_decoder is None:
            hidden_dims_decoder = [128, 64, 32]

        # Encoder
        modules_enc = []
        current_channels_enc = in_channels
        for h_dim in hidden_dims_encoder:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(current_channels_enc, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(32, h_dim//4 if h_dim >=4 else 1), h_dim),
                    nn.SiLU()
                )
            )
            current_channels_enc = h_dim
        self.encoder_conv = nn.Sequential(*modules_enc)
        self.fc_mu = nn.Linear(hidden_dims_encoder[-1] * self.final_encoder_res * self.final_encoder_res, latent_dim * self.final_encoder_res * self.final_encoder_res)


        # Decoder
        self.decoder_input = nn.Linear(latent_dim * self.final_encoder_res * self.final_encoder_res, hidden_dims_decoder[0] * self.final_encoder_res * self.final_encoder_res)
        modules_dec = []
        current_channels_dec = hidden_dims_decoder[0]
        for i in range(len(hidden_dims_decoder) -1):
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels_dec, hidden_dims_decoder[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(min(32, hidden_dims_decoder[i+1]//4 if hidden_dims_decoder[i+1] >=4 else 1), hidden_dims_decoder[i+1]),
                    nn.SiLU()
                )
            )
            current_channels_dec = hidden_dims_decoder[i+1]

        modules_dec.append(
            nn.Sequential(
                nn.ConvTranspose2d(current_channels_dec, current_channels_dec, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(min(32, current_channels_dec//4 if current_channels_dec >=4 else 1), current_channels_dec),
                nn.SiLU(),
                nn.Conv2d(current_channels_dec, in_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        )
        self.decoder_conv = nn.Sequential(*modules_dec)


    def encode(self, image_tensor):
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Input image_tensor must be a torch.Tensor.")
        x = self.encoder_conv(image_tensor)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        return mu.view(image_tensor.size(0), self.latent_dim, self.final_encoder_res, self.final_encoder_res)


    def decode(self, latent_tensor):
        if not isinstance(latent_tensor, torch.Tensor):
            raise TypeError("Input latent_tensor must be a torch.Tensor.")
        x = latent_tensor.view(latent_tensor.size(0), -1)
        x = self.decoder_input(x)
        x = x.view(x.size(0), -1, self.final_encoder_res, self.final_encoder_res) # Use the first dim of hidden_dims_decoder
        return self.decoder_conv(x)

class TimeEmbedding(nn.Module):
    def __init__(self, time_channels, embed_dim, num_groups_time_emb=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.time_channels = time_channels

    def forward(self, t):
        if t.ndim == 0: # if t is a scalar tensor
            t = t.unsqueeze(0)
        if t.ndim == 1 and t.dtype == torch.long: # if t is (batch_size,) of timesteps
             half_dim = self.time_channels // 2
             emb = torch.exp(torch.arange(half_dim, device=t.device) * -(np.log(10000.0) / (half_dim -1)))
             emb = t[:, None] * emb[None, :]
             emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
             if self.time_channels % 2 == 1: # zero pad
                emb = F.pad(emb, (0,1,0,0))
        elif t.ndim == 2 and t.shape[1] == self.time_channels: # if t is already (batch_size, time_channels)
            emb = t
        else:
            raise ValueError(f"Unexpected timestep tensor shape: {t.shape}")
        return self.mlp(emb)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = BasicConvBlock(in_channels, out_channels)
        self.conv2 = BasicConvBlock(out_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_bias = self.time_proj(F.silu(t_emb))
        h = h + time_bias[:, :, None, None]
        h = self.conv2(h)
        return self.downsample(h), h

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = BasicDeconvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = BasicConvBlock(out_channels + skip_channels, out_channels) # Concatenate skip connection
        self.conv2 = BasicConvBlock(out_channels, out_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, skip_x, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        h = self.conv1(x)
        time_bias = self.time_proj(F.silu(t_emb))
        h = h + time_bias[:, :, None, None]
        return self.conv2(h)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, time_channels=32, context_dim=None, base_channels=64, channel_mults=(1, 2, 4)):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_channels, base_channels * 4)
        self.initial_conv = BasicConvBlock(in_channels, base_channels)

        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(UNetDownBlock(current_channels, out_ch, base_channels * 4))
            current_channels = out_ch

        self.bottleneck1 = BasicConvBlock(current_channels, current_channels * 2)
        self.bottleneck2 = BasicConvBlock(current_channels * 2, current_channels)

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            in_ch = base_channels * mult
            skip_ch = in_ch
            out_ch = base_channels * channel_mults[i-1] if i > 0 else base_channels
            self.up_blocks.append(UNetUpBlock(current_channels, skip_ch, out_ch, base_channels * 4))
            current_channels = out_ch

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.context_dim = context_dim 

    def forward(self, noisy_latents, timestep, context=None):
        if not isinstance(noisy_latents, torch.Tensor):
            raise TypeError("Input noisy_latents must be a torch.Tensor.")
        if not isinstance(timestep, (torch.Tensor, int, float)):
             raise TypeError("Input timestep must be a torch.Tensor or int or float.")
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep] * noisy_latents.shape[0], device=noisy_latents.device, dtype=torch.long)


        t_emb = self.time_embedding(timestep)
        x = self.initial_conv(noisy_latents)

        skip_connections = []
        for block in self.down_blocks:
            x, skip_x = block(x, t_emb)
            skip_connections.append(skip_x)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        skip_connections = skip_connections[::-1]
        for i, block in enumerate(self.up_blocks):
            x = block(x, skip_connections[i], t_emb)

        return self.final_conv(x)

class DiffusionScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="linear"):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2 # Improved linear schedule
        elif beta_schedule == "cosine":
            s = 0.008
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999) # Clip to avoid issues
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)


        self.num_inference_steps = None
        self.timesteps = None
        self.init_noise_sigma = 1.0

    def _get_variance(self, t, predicted_variance=None, variance_type="fixed_small"):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        variance = torch.clamp(variance, min=1e-20)
        return variance


    def set_timesteps(self, num_inference_steps, device='cpu'):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(self, original_samples, noise, timesteps):
        if not isinstance(original_samples, torch.Tensor) or not isinstance(noise, torch.Tensor):
            raise TypeError("original_samples and noise must be torch.Tensors.")
        if not isinstance(timesteps, (torch.Tensor, int)):
            raise TypeError("timesteps must be a torch.Tensor or int.")

        if isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps], device=original_samples.device, dtype=torch.long)

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator=None, return_dict: bool = True):
        t = timestep
        if isinstance(t, torch.Tensor): t = t.item()

        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=sample.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        pred_epsilon = model_output

        posterior_mean_coef1 = self.posterior_mean_coef1[t].to(sample.device)
        posterior_mean_coef2 = self.posterior_mean_coef2[t].to(sample.device)

        if t == 0:
            pred_prev_sample = pred_original_sample 
        else:
            pred_prev_sample_mean = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
            variance = self._get_variance(t)
            std_dev_t = variance ** (0.5)
            noise = torch.randn_like(sample, generator=generator) if generator is not None else torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample_mean + std_dev_t * noise
        
        if not return_dict:
            return (pred_prev_sample,)
        return {"prev_sample": pred_prev_sample, "pred_original_sample": pred_original_sample}


class TextEncoder(nn.Module):
    def __init__(self, model_name="placeholder_clip", vocab_size=49408, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def encode_prompt(self, prompt_texts, device='cpu', max_length=77):
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
        
        batch_size = len(prompt_texts)
        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_length), device=device)
        embeddings = self.token_embedding(input_ids) # (batch_size, max_length, embed_dim)
        return torch.zeros(batch_size, max_length, self.embed_dim, device=device, dtype=embeddings.dtype)


class LatentDiffusionPipeline:
    def __init__(self, vae: VAE, unet: UNet, scheduler: DiffusionScheduler, text_encoder: TextEncoder = None, device='cpu'):
        self.vae = vae.to(device).eval()
        self.unet = unet.to(device).eval()
        self.scheduler = scheduler
        self.text_encoder = text_encoder.to(device).eval() if text_encoder else None
        self.device = device
        self.vae_scale_factor = 0.18215 # scale factor for Stable Diffusion VAE

    @torch.no_grad()
    def generate(self,
                 prompt: str = None,
                 negative_prompt: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 generator: torch.Generator = None,
                 initial_latents: torch.Tensor = None,
                 batch_size: int = 1,
                 height: int = 512,
                 width: int = 512,
                 output_type: str = "pil",
                 eta: float = 0.0): # For DDIM-like schedulers, not used in this simplified DDPM step

        do_classifier_free_guidance = guidance_scale > 1.0 and prompt is not None and self.text_encoder is not None

        if prompt is not None and self.text_encoder is None:
            raise ValueError("Text prompt provided but no text_encoder is configured.")

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        latent_channels = self.unet.initial_conv.conv.in_channels # Get from UNet's first layer
        latents_height = height // 8
        latents_width = width // 8

        if initial_latents is None:
            shape = (batch_size, latent_channels, latents_height, latents_width)
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.unet.initial_conv.conv.weight.dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            if initial_latents.shape != (batch_size, latent_channels, latents_height, latents_width):
                raise ValueError("Provided initial_latents have incorrect shape.")
            latents = initial_latents.to(self.device, dtype=self.unet.initial_conv.conv.weight.dtype)

        text_embeddings = None
        if do_classifier_free_guidance:
            if not isinstance(prompt, list):
                prompt = [prompt] * batch_size
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * batch_size
            
            max_length = 77 # Typical for CLIP
            prompt_embeds = self.text_encoder.encode_prompt(prompt, device=self.device, max_length=max_length)
            uncond_embeds = self.text_encoder.encode_prompt(negative_prompt, device=self.device, max_length=max_length)
            text_embeddings = torch.cat([uncond_embeds, prompt_embeds])


        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            # Some schedulers require scaling the input
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 

            noise_pred = self.unet(latent_model_input, t, context=text_embeddings if do_classifier_free_guidance else None)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_output = self.scheduler.step(noise_pred, t, latents, generator=generator)
            latents = step_output["prev_sample"]


        latents = 1.0 / self.vae_scale_factor * latents
        image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            from PIL import Image
            image_np = image.cpu().permute(0, 2, 3, 1).float().numpy() # Ensure float before multiplying by 255
            images = [(Image.fromarray((img * 255).round().astype(np.uint8))) for img in image_np]
            return images if batch_size > 1 else images[0]
        elif output_type == "tensor":
            return image
        else:
            raise ValueError(f"Unsupported output_type: {output_type}.")
