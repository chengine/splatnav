"""
GEMSPLAT configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from gemsplat.data.gemsplat_datamanager import GEMSPLATDataManagerConfig
from gemsplat.gemsplat import SemanticGaussianSplattingModelConfig
from gemsplat.gemsplat_pipeline import GEMSPLATPipelineConfig

from gemsplat.encoders.maskclip_encoder import MaskCLIPNetworkConfig

gemsplat_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gemsplat",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 2},
        
        pipeline=GEMSPLATPipelineConfig(
            datamanager=GEMSPLATDataManagerConfig(
                dataparser=ColmapDataParserConfig(load_3D_points=True,
                )
            ),
            model=SemanticGaussianSplattingModelConfig(
            ),
            #  You can swap the type of input encoder by specifying different clip_model_types, e.g., "RN50x64," "ViT-B/16."
            network=MaskCLIPNetworkConfig(
                clip_model_type="RN50x64", clip_n_dims=512
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "color": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=30000,
                ),
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            "clip_embeds": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=30000,
                ),
            },
            "clip_encoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "clip_decoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer", #"viewer_beta",
    ),
    description="Config for GEMSPLAT",
)
