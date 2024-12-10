import numpy as np
import torch
from typing import List, Tuple, Optional

def get_goal_locations(
    nerf,
    goal_queries,
    negatives: str = "object, things, stuff, texture",
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
):
    print("Setting the Goal Locations...")

    # option to use bounding box
    use_bounding_box = bounding_box_min is not None and bounding_box_min is not None

    # generate the point cloud of the environment
    env_pcd, _, env_attr = nerf.generate_point_cloud(
        use_bounding_box=use_bounding_box,
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
    )

    # goal locations
    goal_locations = []

    for query in goal_queries:
        print('Query', query)

        # get the semantic outputs
        semantic_info = nerf.get_semantic_point_cloud(
            positives=query, negatives=negatives, pcd_attr=env_attr
        )

        # scaled similarity
        sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
        sc_sim = sc_sim / (sc_sim.max() + 1e-6)

        # get the maximizer
        goal_pt = np.asarray(env_pcd.points)[sc_sim.argmax()]

        # transform from the nerfframe to the dataframe
        goal_pt = torch.tensor(goal_pt, device=nerf.device).float()

        # NOTE: This is specific to our setup. Since the scene is NED, we add some z-height to float above the goal #
        goal_pt[2] -= 0.4

        goal_locations.append(goal_pt[:3].cpu().numpy().astype(np.float32))

    return goal_locations