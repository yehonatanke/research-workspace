import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model implementation.

    Args:
        image_size (int): Size of the input image (assumed square). Default: 224.
        patch_size (int): Size of the patches to divide the image into. Default: 16.
        num_classes (int): Number of output classes for classification. Required.
        dim (int): Embedding dimension for patches and transformer layers. Default: 768.
        depth (int): Number of transformer encoder layers. Default: 12.
        heads (int): Number of attention heads in the multi-head attention mechanism. Default: 12.
        mlp_dim (int): Dimension of the feed-forward network within transformer blocks. Default: 3072.
        channels (int): Number of input image channels. Default: 3 (for RGB).
        dropout (float): Dropout rate for the MLP head. Default: 0.1.
        emb_dropout (float): Dropout rate for the patch and position embeddings. Default: 0.1.
    """

    def __init__(self, *, image_size=224, patch_size=16, num_classes=8, dim=768, depth=12, heads=12, mlp_dim=3072, channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        # Parameter Validation 
        if image_size % patch_size != 0:
            raise ValueError('Image dimensions must be divisible by the patch size.')

        # Calculate Dimensions 
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # Learnable Parameters
        # Class token (prepended to patch embeddings)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Positional embedding (learnable, includes class token position)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        #  Layers 
        # Patch embedding
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        # Dropout for embeddings
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder
        # Create a single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,  # Input feature dimension
            nhead=heads,  # Number of attention heads
            dim_feedforward=mlp_dim,  # Dimension in the feed-forward network
            dropout=dropout,  # Dropout rate within the transformer layer
            activation=F.gelu,  # Activation function (GELU is common in Transformers)
            batch_first=True,  # Expect input as (batch, seq, features)
            norm_first=True  # Apply LayerNorm before attention/FFN (Pre-LN) - often more stable
        )
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth  # Number of layers in the stack
        )

        # MLP Head for Classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # Normalize the features of the class token
            nn.Linear(dim, num_classes)  # Linear layer to map to output classes
        )

        #  Store config 
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, img):
        """
        Forward pass of the ViT model.

        Args:
            img (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).
                                Expected height and width are `image_size`.

        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, num_classes).
        """
        b, c, h, w = img.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Input image size ({h}x{w}) doesn't match model's expected size ({self.image_size}x{self.image_size})")

        # 1. Patch Embedding
        # Input: (b, c, h, w) = (b, 3, 224, 224)
        # Output: (b, dim, h//p, w//p) = (b, 768, 14, 14)
        x = self.patch_embedding(img)

        # Flatten spatial dimensions and rearrange for transformer
        # Output: (b, dim, num_patches) -> (b, num_patches, dim)
        x = x.flatten(2).transpose(1, 2)  # (b, 196, 768) where 196 = (224/16)**2
        _, n, _ = x.shape  # n = num_patches

        # 2. Prepend Class Token
        # Expand class token to match batch size
        # cls_tokens shape: (b, 1, dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # Concatenate class token and patch embeddings
        # Output: (b, num_patches + 1, dim) = (b, 197, 768)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add Positional Embedding
        # Output: (b, num_patches + 1, dim)
        x += self.pos_embedding[:, :(n + 1)]

        # 4. Apply Dropout
        x = self.dropout(x)

        # 5. Pass through Transformer Encoder
        # Input/Output: (b, num_patches + 1, dim)
        x = self.transformer_encoder(x)

        # 6. Extract Class Token Output
        # We only use the output corresponding to the class token for classification
        # Output: (b, dim)
        cls_token_output = x[:, 0]  # Select the first token's output

        # 7. Pass through MLP Head
        # Input: (b, dim)
        # Output: (b, num_classes)
        logits = self.mlp_head(cls_token_output)

        return logits

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)

    def get_history(self, metric_name: str):
        return self.history.get(metric_name, [])

    def get_all_metrics(self):
        return self.history
