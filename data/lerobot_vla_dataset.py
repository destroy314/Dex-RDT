import numpy as np
from pathlib import Path
import yaml
import json
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from .bson_vla_dataset import plot_distributions


class LeRobotVLADataset:
    """
    Simple wrapper around LeRobotDataset with key mapping to match BsonVLADataset interface.
    """
    
    def __init__(self, repo_dir: str="data/ours/true/output/airbot_dexterous_bimanual_dexterous_manipulation", 
                 sub_sample=1.0, normalize_mode: str="min_max", stats_file: str="v4_lerobot_stats/dataset_statistics.json") -> None:
        # Load config
        config_path = Path("configs/base.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.camera_keys = [
            ("camera_high", "cam_high"),
            ("camera_left", "cam_left_wrist"),
            ("camera_right", "cam_right_wrist"),
            ("camera_front", "cam_third_view"),
        ]
        self.DATASET_NAME = "ours"
        
        # Normalization settings
        self.normalize_mode = normalize_mode
        self.stats = None
        if normalize_mode and stats_file:
            self._load_statistics(stats_file)

        with open(os.path.join(repo_dir, "meta", "episodes_stats.jsonl"), 'r') as f:
            episodes = [json.loads(line) for line in f]
            self.tasks = {ep['episode_index']: ep['instruction'] for ep in episodes}

        # repo_ids = []
        # for dataset_item in os.listdir(repo_dir):
        #     dataset_path = os.path.join(repo_dir, dataset_item)
        #     if os.path.exists(os.path.join(dataset_path, "meta")):
        #         repo_ids.append(dataset_item)
        
        # metadata = LeRobotDatasetMetadata(repo_ids[0], repo_dir)
        metadata = LeRobotDatasetMetadata("", repo_dir)
        fps = metadata.fps

        delta_timestamps = {
            'states': [0],
            "actions": [i/fps for i in range(self.CHUNK_SIZE)]
        }
        for cam,_ in self.camera_keys:
            delta_timestamps[f"observation.images.{cam}"] = [i/fps for i in range(1-self.IMG_HISTORY_SIZE, 1)]

        # self.dataset = MultiLeRobotDataset(repo_ids, repo_dir, delta_timestamps=delta_timestamps)
        self.dataset = LeRobotDataset("", repo_dir, delta_timestamps=delta_timestamps)

    def __len__(self):
        return len(self.dataset)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """
        Get a training sample with key mapping to match BsonVLADataset format.
        
        Args:
            index (int, optional): The index of the sample. If None, a random sample is chosen.
            state_only (bool, optional): If True, returns the full state/action trajectories.
                                         Otherwise, returns a single timestep sample.
        Returns:
           A dictionary containing the training sample.
        """
        if state_only:
            return self._get_state_only_item(index)
        
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        
        item = self.dataset[index]

        ep_idx = item["episode_index"].item()
        task = self.tasks[ep_idx]
        # Map keys to BsonVLADataset format
        sample = {
            "meta": {
                "dataset_name": self.DATASET_NAME,
                'episode_idx': item['episode_index'].item(),
                'step_id': item['frame_index'].item(),
                'instruction': task,
            },
            "state": item['states'].numpy(),
            "actions": item['actions'].numpy(),
            "state_indicator": np.ones(item["states"].shape[1], dtype=bool),
        }
        
        sample["state_std"] = np.std(sample["state"], axis=0)
        sample["state_mean"] = np.mean(sample["state"], axis=0)
        sample["state_norm"] = np.sqrt(np.mean(sample["state"]**2, axis=0))
        
        # Apply normalization if enabled
        if self.normalize_mode:
            sample["state"] = self._normalize_data(sample["state"], 'state')
            sample["actions"] = self._normalize_data(sample["actions"], 'action')
        
        for key, out_key in self.camera_keys:
            image = item[f"observation.images.{key}"]
            sample[out_key] = (image.unsqueeze(0).numpy().transpose((0,2,3,1))*255).astype(np.uint8) # 仅支持单帧
            sample[out_key + '_mask'] = np.ones(self.IMG_HISTORY_SIZE, dtype=bool)
        
        return sample
    
    def _get_state_only_item(self, index: int = None):
        """
        Get single sample state and action for statistics collection.
        For efficiency, just return individual samples rather than full episodes.
        """
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        
        try:
            # Get single sample - much more efficient for statistics
            item = self.dataset[index]
            
            # Extract state and action from single sample
            state = item["states"].numpy()
            action = item["actions"].numpy()
            
            # Apply normalization if enabled
            if self.normalize_mode:
                state = self._normalize_data(state, 'state')
                action = self._normalize_data(action, 'action')
            
            return {
                "state": state,
                "action": action
            }
        except Exception as e:
            # Ignore file not found or other errors, return None
            return None
    
    def _load_statistics(self, stats_file: str):
        """
        Load statistics from JSON file for normalization.
        
        Args:
            stats_file (str): Path to dataset_statistics.json file
        """
        try:
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded statistics from {stats_file}")
            print(f"State dim: {len(self.stats['state']['mean'])}, Action dim: {len(self.stats['action']['mean'])}")
        except Exception as e:
            print(f"Warning: Failed to load statistics from {stats_file}: {e}")
            self.stats = None
    
    def _normalize_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Normalize data using loaded statistics.
        
        Args:
            data (np.ndarray): Data to normalize (state or action)
            data_type (str): 'state' or 'action'
            
        Returns:
            np.ndarray: Normalized data
        """
        if self.stats is None or self.normalize_mode is None:
            return data
            
        if data_type not in self.stats:
            print(f"Warning: No statistics found for {data_type}")
            return data
            
        stats_data = self.stats[data_type]
        
        if self.normalize_mode == 'mean_std':
            # Normalize using mean and standard deviation: (x - mean) / std
            mean = np.array(stats_data['mean'])
            std = np.array(stats_data['std'])
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            normalized = (data - mean) / std
            
        elif self.normalize_mode == 'min_max':
            # Normalize using percentiles as min/max: (x - min) / (max - min)
            min_val = np.array(stats_data['percentile_1'])
            max_val = np.array(stats_data['percentile_99'])
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            
        else:
            print(f"Warning: Unknown normalization mode {self.normalize_mode}")
            return data
            
        return normalized


def collect_statistics(dataset: LeRobotVLADataset, num_samples=10000):
    """
    Collect statistics for state and action dimensions using random sampling.
    Much more efficient than the BSON version since we don't reconstruct full episodes.
    
    Args:
        dataset: LeRobotVLADataset instance
        num_samples: Maximum number of samples to collect statistics from
        
    Returns:
        dict: Statistics containing mean, std, 1st and 99th percentiles
    """
    print(f"Collecting statistics from random samples (target: {num_samples} samples)...")
    
    states = []
    actions = []
    total_samples = 0
    failed_samples = 0
    
    # Random sampling approach with progress bar
    pbar = tqdm(total=num_samples, desc="Collecting samples")
    
    while total_samples < num_samples:
        try:
            # Get random sample
            sample = dataset.get_item(state_only=True)
            
            if sample is None:
                failed_samples += 1
                pbar.set_postfix({"failed": failed_samples})
                continue
                
            # Extract state and action from single sample
            state = sample['state']  # Shape: (1, state_dim)
            action = sample['action']  # Shape: (1, action_dim)
            
            for t in range(len(state)):
                states.append(state[t])
            for t in range(len(action)):
                actions.append(action[t])
            total_samples += 1
            pbar.update(1)
            pbar.set_postfix({"failed": failed_samples})
                
        except Exception as e:
            failed_samples += 1
            pbar.set_postfix({"failed": failed_samples})
            # Ignore errors (file not found, etc.)
            continue
    
    pbar.close()
    
    if not states or not actions:
        raise ValueError("No valid samples collected")
    
    # Convert to numpy arrays
    states = np.array(states)  # Shape: (num_samples, state_dim)
    actions = np.array(actions)  # Shape: (num_samples, action_dim)
    
    print(f"Collected {len(states)} valid samples, failed: {failed_samples}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Calculate statistics
    stats = {
        'state': {
            'mean': np.mean(states, axis=0).tolist(),
            'std': np.std(states, axis=0).tolist(),
            'percentile_1': np.percentile(states, 1, axis=0).tolist(),
            'percentile_99': np.percentile(states, 99, axis=0).tolist()
        },
        'action': {
            'mean': np.mean(actions, axis=0).tolist(),
            'std': np.std(actions, axis=0).tolist(),
            'percentile_1': np.percentile(actions, 1, axis=0).tolist(),
            'percentile_99': np.percentile(actions, 99, axis=0).tolist()
        },
        'metadata': {
            'num_samples': len(states),
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
            'failed_samples': failed_samples,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return stats, states, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LeRobot VLA Dataset Testing and Statistics')
    parser.add_argument('--stat', action='store_true', help='Collect statistics from 10k samples')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples for statistics (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='lerobot_stats', help='Output directory for statistics and plots')
    parser.add_argument('--normalize', choices=['mean_std', 'min_max'], help='Normalization mode')
    parser.add_argument('--stats_file', type=str, help='Path to dataset_statistics.json for normalization')
    parser.add_argument('--repo_dir', type=str, default='data/ours/true/output/airbot_dexterous_bimanual_dexterous_manipulation', help='Repository directory')
    
    args = parser.parse_args()
    
    # --- Dataset Initialization ---
    ds = LeRobotVLADataset(repo_dir=args.repo_dir, normalize_mode=args.normalize, stats_file=args.stats_file)
    
    if len(ds) == 0:
        print("\nDataset initialized but contains no valid episodes.")
        exit(1)
    
    if args.stat:
        # Collect statistics
        try:
            ds.dataset.meta.info["features"] = {}
            object.__setattr__(ds.dataset, 'image_transforms', None)
            stats, states, actions = collect_statistics(ds, args.num_samples)
            
            # Save statistics to JSON
            os.makedirs(args.output_dir, exist_ok=True)
            stats_file = os.path.join(args.output_dir, 'dataset_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStatistics saved to: {stats_file}")
            
            # Generate distribution plots
            plot_distributions(states, actions, args.output_dir)
            
            # Print summary
            print("\n=== Statistics Summary ===")
            print(f"State dimensions: {stats['metadata']['state_dim']}")
            print(f"Action dimensions: {stats['metadata']['action_dim']}")
            print(f"Samples collected: {stats['metadata']['num_samples']}")
            print(f"\nState statistics:")
            print(f"  Mean range: [{np.min(stats['state']['mean']):.4f}, {np.max(stats['state']['mean']):.4f}]")
            print(f"  Std range: [{np.min(stats['state']['std']):.4f}, {np.max(stats['state']['std']):.4f}]")
            print(f"\nAction statistics:")
            print(f"  Mean range: [{np.min(stats['action']['mean']):.4f}, {np.max(stats['action']['mean']):.4f}]")
            print(f"  Std range: [{np.min(stats['action']['std']):.4f}, {np.max(stats['action']['std']):.4f}]")
            
        except Exception as e:
            print(f"Error collecting statistics: {e}")
            exit(1)
    else:
        # --- Example Usage ---
        print(f"\n--- Testing get_item (state_only=False) for one item ---")
        sample = ds.get_item()
        print("Sample keys:", sample.keys())
        print("Meta:", sample['meta'])
        print("State shape:", sample['state'].shape)
        print("State indicator shape:", sample['state_indicator'].shape)
        print("State mean shape:", sample['state_mean'].shape)
        print("Actions shape:", sample['actions'].shape)
        
        # Print camera info
        camera_keys = [k for k in sample.keys() if k.startswith('cam_') and not k.endswith('_mask')]
        for cam_key in camera_keys:
            print(f"Cam {cam_key} shape:", sample[cam_key].shape)
        
        print(f"\n--- Testing get_item (state_only=True) for one item ---")
        try:
            state_sample = ds.get_item(state_only=True)
            if state_sample:
                print("State sample keys:", state_sample.keys())
                print("Full state trajectory shape:", state_sample['state'].shape)
                print("Full action trajectory shape:", state_sample['action'].shape)
            else:
                print("No state-only sample returned")
        except Exception as e:
            print(f"Error getting state-only sample: {e}")

        print("\n--- Testing get_item for 10 items ---")
        for i in range(10):
            try:
                sample = ds.get_item()
                print(f"Sample {i}: state {sample['state'].shape}, actions {sample['actions'].shape}")
            except Exception as e:
                print(f"Error getting sample {i}: {e}")
