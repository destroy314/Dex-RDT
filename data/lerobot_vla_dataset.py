import numpy as np
from pathlib import Path
import yaml
import os

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset


class LeRobotVLADataset:
    """
    Simple wrapper around LeRobotDataset with key mapping to match BsonVLADataset interface.
    """
    
    def __init__(self, repo_dir: str, sub_sample=1.0) -> None:
        # Load config
        config_path = Path("configs/base.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.camera_keys = [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
            "cam_third_view",
        ]
        self.DATASET_NAME = "ours"

        repo_ids = []
        for dataset_item in os.listdir(repo_dir):
            dataset_path = os.path.join(repo_dir, dataset_item)
            if os.path.exists(os.path.join(dataset_path, "meta")):
                repo_ids.append(dataset_item)
        
        metadata = LeRobotDatasetMetadata(repo_ids[0], repo_dir)
        fps = metadata.fps

        delta_timestamps = {
            'observation.state': 0,
            "action": [i/fps for i in range(self.CHUNK_SIZE)]
        }
        for cam in self.camera_keys:
            delta_timestamps[f"observation.images.{cam}"] = [i/fps for i in range(-self.IMG_HISTORY_SIZE, 0, -1)]

        self.dataset = MultiLeRobotDataset(repo_ids, repo_dir, delta_timestamps=delta_timestamps)

    def __len__(self):
        return len(self.dataset)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """
        Get a training sample with key mapping to match BsonVLADataset format.
        """
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        
        item = self.dataset[index]
        
        if state_only:
            raise NotImplementedError
        
        # Map keys to BsonVLADataset format
        sample = {
            "meta": {
                'episode_idx': item['episode_index'].item(),
                'step_id': item['frame_index'].item(),
                'task': item.get('task', 'do something')
            },
            "state": item['observation.state'].numpy(),
            "actions": item['action'].numpy(),
            "state_indicator": np.ones(item['observation.state'].shape[0], dtype=bool),
        }
        
        # Add normalization stats (simplified - use dataset stats)
        if hasattr(self.dataset, 'stats') and 'observation.state' in self.dataset.stats:
            stats = self.dataset.stats['observation.state']
            sample["state_std"] = stats['std']
            sample["state_mean"] = stats['mean']
            sample["state_norm"] = (sample["state"] - sample["state_mean"]) / (sample["state_std"] + 1e-8)
        else:
            sample["state_std"] = np.ones_like(sample["state"])
            sample["state_mean"] = np.zeros_like(sample["state"])
            sample["state_norm"] = sample["state"]
        
        for key in self.camera_keys:
            if key in item:
                image = item[f"observation.images.{key}"]
                sample[key] = image
                sample[key + '_mask'] = np.ones(self.IMG_HISTORY_SIZE, dtype=bool)
        
        return sample


if __name__ == "__main__":
    try:
        ds = LeRobotVLADataset(repo_dir="data/ours_lerobot")
        
        if len(ds) > 0:
            print(f"\n--- Testing get_item (state_only=False) ---")
            sample = ds.get_item()
            print("Sample keys:", sample.keys())
            print("Meta:", sample['meta'])
            print("State shape:", sample['state'].shape)
            print("Actions shape:", sample['actions'].shape)
            if 'cam_high' in sample:
                print("Cam High shape:", sample['cam_high'].shape)

            print(f"\n--- Testing get_item (state_only=True) ---")
            state_sample = ds.get_item(state_only=True)
            print("State sample keys:", state_sample.keys())
            print("Full state trajectory shape:", state_sample['state'].shape)
            print("Full action trajectory shape:", state_sample['action'].shape)
        else:
            print("\nDataset is empty.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Please specify a valid LeRobot dataset repo_id")
