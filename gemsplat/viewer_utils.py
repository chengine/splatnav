from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from nerfstudio.viewer.server.viewer_elements import *
from nerfstudio.utils.rich_utils import CONSOLE
from functools import cached_property
from gemsplat.encoders.image_encoder import BaseImageEncoder

@dataclass
class ViewerUtils:
    image_encoder: BaseImageEncoder
    pca_proj: Optional[torch.Tensor] = None
    low_rank_min: Optional[torch.Tensor] = None
    low_rank_max: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def clip(self):
        from gemsplat.clip.clip import load
        from gemsplat.encoders.maskclip_encoder import MaskCLIPNetworkConfig

        model, _ = load(self.image_encoder.config.clip_model_type, device=self.device)
                
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        from gemsplat.clip.clip import tokenize

        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        if is_positive:
            self.positives = texts
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed /= embed.norm(dim=-1, keepdim=True)
            
            self.pos_embed = embed
        else:
            self.negatives = texts
            # We don't average the negatives as we compute pair-wise softmax
            embed /= embed.norm(dim=-1, keepdim=True)
            self.neg_embed = embed    

    @property
    def has_positives(self) -> bool:
        return self.positives and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return self.negatives and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")

