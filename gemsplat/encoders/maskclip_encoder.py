from dataclasses import dataclass, field
from typing import Tuple, Type, List

import torch
import torchvision

import gc

from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from gemsplat.clip import clip
from gemsplat.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig


@dataclass
class MaskCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: MaskCLIPNetwork)
    clip_model_type: str =  "ViT-B/32"
    clip_n_dims: int = 512
    # negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    skip_center_crop: bool = True
    batch_size: int = 1

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.clip_model_type,
            "skip_center_crop": cls.skip_center_crop,
        }

class MaskCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: MaskCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.im_h = None
        self.im_w = None
        self.model, self.preprocess = clip.load(self.config.clip_model_type)
        self.model.eval()
        self.model.to("cuda")

        # self.tokenizer = clip.tokenize
        # self.clip_n_dims = self.config.clip_n_dims

        # self.positives = ["hand sanitizer"]
        # self.negatives = self.config.negatives
        # with torch.no_grad():
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
        #     self.pos_embeds = self.model.encode_text(tok_phrases)
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
        #     self.neg_embeds = self.model.encode_text(tok_phrases)
        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        # self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # assert (
        #     self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        # ), "Positive and negative embeddings must have the same dimensionality"
        # assert (
        #     self.pos_embeds.shape[1] == self.clip_n_dims
        # ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "clip_openai_{}".format(self.config.clip_model_type)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def set_positives(self, text_list):
        pass

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        pass
    
    def encode_image(self, image_list):
        # Patch the preprocess if we want to skip center crop
        if MaskCLIPNetworkConfig.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in self.preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            self.preprocess = Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, CenterCrop)]
            )
            print("Skipping center crop")

        # Preprocess the images
        # images = [Image.open(path) for path in image_paths]
        rgb_image_transform = torchvision.transforms.ToPILImage()
        images = [rgb_image_transform(image_list[i,:,:,:]).convert("RGB") for i in range(image_list.size()[0])]
        preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        preprocessed_images = preprocessed_images.to("cuda")  # (b, 3, h, w)
        print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")

        # Get CLIP embeddings for the images
        embeddings = []
        
        with torch.no_grad():
        
            for i in tqdm(
                range(0, len(preprocessed_images), 
                    MaskCLIPNetworkConfig.batch_size),
                desc="Extracting CLIP features",
            ):
                batch = preprocessed_images[i : i + MaskCLIPNetworkConfig.batch_size]
                embeddings.append(self.model.get_patch_encodings(batch))

        embeddings = torch.cat(embeddings, dim=0)

        # Reshape embeddings from flattened patches to patch height and width
        h_in, w_in = preprocessed_images.shape[-2:]
        if self.config.clip_model_type.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
        elif self.config.clip_model_type.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            h_out, w_out = int(h_out), int(w_out)
        else:
            raise ValueError(f"Unknown CLIP model name: {MaskCLIPNetworkConfig.clip_model_type}")
        self.embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
        print(f"Extracted CLIP embeddings of shape {embeddings.shape}")

        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = self.embeddings.shape[1:3]
        assert len(self.im_h) == 1, "All images must have the same height"
        assert len(self.im_w) == 1, "All images must have the same width"
        self.im_h, self.im_w = self.im_h.pop(), self.im_w.pop()
        self.scale_h = feat_h / self.im_h
        self.scale_w = feat_w / self.im_w

        # Delete and clear memory to be safe
        del self.model
        del self.preprocess
        del preprocessed_images
        torch.cuda.empty_cache()
        gc.collect()

        return self.embeddings
