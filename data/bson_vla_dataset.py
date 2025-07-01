import os
import fnmatch
import yaml
import numpy as np
import bson
import av
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import re
import pathlib

class EpisodeInfo:
    """Custom class to store episode path and action text"""
    def __init__(self, path, action_text):
        self.path = path
        self.action = action_text
    
    def __str__(self):
        return self.path
    
    def __repr__(self):
        return f"EpisodeInfo(path={self.path}, action={self.action[:20]}...)" if len(self.action) > 20 else f"EpisodeInfo(path={self.path}, action={self.action})"


class BsonVLADataset:
    """
    Modified BsonVLADataset to handle new directory-based episode structure with:
    - Main BSON file for arm data
    - Separate xhand_control_data.bson for dexterous hand data
    - Image file sequences from multiple cameras
    """
    def __init__(self, bson_dir: str="data/ours", sub_sample=1.0) -> None:
        """
        Initializes the BsonVLADataset.

        Args:
            bson_dir (str): The path to the dataset directory containing episode folders.
            sub_sample (float): The fraction of the dataset to use.
        """
        self.DATASET_NAME = "ours"
        
        self.ext_image_keys = ["camera_0", "camera_3", "camera_8"]
        self.ext_image_names = ["cam_left_wrist", "cam_third_view", "cam_right_wrist"]
        self.image_keys = ["head_camera"] + self.ext_image_keys

        print("Finding episode directories...")
        self.episode_infos: list[EpisodeInfo] = []
        
        # Find all episode directories in the new structure
        for action_item in os.listdir(bson_dir):
            action_path = os.path.join(bson_dir, action_item)
            if os.path.isdir(action_path) and action_item.startswith('action'):
                # Read action.txt content
                action_txt_path = os.path.join(action_path, "action.txt")
                if os.path.exists(action_txt_path):
                    with open(action_txt_path, 'r') as f:
                        action_text = f.read().strip()
                else:
                    print(f"Warning: Missing action.txt in {action_path}")
                    action_text = ""
                
                # Look for episode subdirectories within each action directory
                for episode_item in os.listdir(action_path):
                    episode_info = os.path.join(action_path, episode_item)
                    if os.path.isdir(episode_info) and episode_item.startswith('episode'):
                        # Check if required files exist in the episode subdirectory
                        main_bson = os.path.join(episode_info, "episode_0.bson")
                        xhand_bson = os.path.join(episode_info, "xhand_control_data.bson")
                        
                        if os.path.exists(main_bson) and os.path.exists(xhand_bson):
                            # Store episode info with action text
                            episode_info = EpisodeInfo(episode_info, action_text)
                            self.episode_infos.append(episode_info)
                        else:
                            print(f"Warning: Missing required files in {episode_info}")
        
        # Sort by episode path for consistency
        self.episode_infos.sort(key=lambda x: x.path)
        print(f"Found {len(self.episode_infos)} valid episode directories.")

        # Sub-sampling logic from original
        indices = np.arange(len(self))
        np.random.seed(42)
        np.random.shuffle(indices)

        superior = sub_sample > 0.5
        if superior:
            sub_sample = 1 - sub_sample
        split = int(len(indices) * sub_sample)
        if superior:
            indices = indices[split:]
        else:
            indices = indices[:split]
        self.episode_infos = [self.episode_infos[i] for i in indices]
        print(f"Using {len(self.episode_infos)} episodes after subsampling.")

        # Load config from YAML
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']

        self._video_cache = {}
        self._image_cache = {}
        
        # Get each episode's len to calculate sample weights
        print("Pre-calculating episode lengths for sampling...")
        episode_lens = []
        valid_episode_infos = []
        for episode_info in self.episode_infos:
            valid, res = self.parse_episode_state_only(episode_info)
            if valid:
                _len = res['state'].shape[0]
                episode_lens.append(_len)
                valid_episode_infos.append(episode_info)
            else:
                print(f"Skipping invalid or too short episode: {episode_info}")
        
        self.episode_infos = valid_episode_infos
        if not episode_lens:
            raise ValueError("No valid episodes found in the provided directory.")
             
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
        print("Dataset initialized.")

    def __len__(self):
        return len(self.episode_infos)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """
        Get a training sample.

        Args:
            index (int, optional): The index of the episode. If None, a random episode is chosen.
            state_only (bool, optional): If True, returns the full state/action trajectories.
                                         Otherwise, returns a single timestep sample.
        Returns:
           A dictionary containing the training sample.
        """
        while True:
            if index is None:
                # Sample an episode based on its length
                episode_info = np.random.choice(self.episode_infos, p=self.episode_sample_weights)
            else:
                episode_info = self.episode_infos[index]

            parser = self.parse_episode if not state_only else self.parse_episode_state_only
            valid, sample = parser(episode_info)
            
            if valid:
                return sample
            else:
                # If randomly chosen episode was invalid, try another random one
                if index is None:
                    print(f"Warning: Invalid sample from {episode_info}, resampling...")
                    continue
                # If specific index is invalid, it's an issue with that file
                else:
                    raise RuntimeError(f"Episode at index {index} ({episode_info}) is invalid.")

    def _extract_data_from_episode(self, episode_info: EpisodeInfo) -> Optional[Dict]:
        """
        Extracts numerical data from both main BSON and xhand BSON files,
        and prepares image paths for later loading.
        """
        # Handle EpisodeInfo object or string path
        path = episode_info.path
        
        main_bson_path = os.path.join(path, "episode_0.bson")
        xhand_bson_path = os.path.join(path, "xhand_control_data.bson")
        
        # Load main BSON (arm data)
        try:
            with open(main_bson_path, 'rb') as f:
                main_bson_content = bson.decode(f.read())["data"]
        except Exception as e:
            print(f"Error reading main BSON file {main_bson_path}: {e}")
            return None

        # Load xhand BSON
        try:
            with open(xhand_bson_path, 'rb') as f:
                xhand_data = bson.decode(f.read())
        except Exception as e:
            print(f"Error reading xhand BSON file {xhand_bson_path}: {e}")
            return None

        # Define data keys for arms (same as original)
        arm_dim, eef_dim = 6, 12
        keys = {
            "left_obs_arm": "/observation/left_arm/joint_state",
            "right_obs_arm": "/observation/right_arm/joint_state",
            # "left_act_arm": "/action/left_arm/joint_state",
            # "right_act_arm": "/action/right_arm/joint_state",
        }

        # Check frame counts
        arm_frame_num = len(main_bson_content[keys["left_obs_arm"]])
        xhand_frame_num = len(xhand_data['frames'])
        
        if arm_frame_num == 0 or xhand_frame_num == 0:
            return None
        
        # Use minimum frame count to ensure alignment
        frame_num = min(arm_frame_num, xhand_frame_num)
        
        # Extract arm data
        state = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)
        action = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)
        
        for i in range(frame_num):
            state[i, :] = np.concatenate([
                main_bson_content[keys["left_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["left_hand"],
                main_bson_content[keys["right_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["right_hand"]
            ])
            # As per requirement: use observation as action for arms
            action[i, :] = np.concatenate([
                main_bson_content[keys["left_obs_arm"]][i]["data"]["pos"], # TODO use obs as act for now
                xhand_data['frames'][i]["action"]["left_hand"],
                main_bson_content[keys["right_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["action"]["right_hand"]
            ])
        
        # TODO not used?
        # Extract image data info
        images_info = {}
        
        # Head camera from BSON (original video stream)
        images_info['head_camera'] = main_bson_content.get("/images/head_camera")
        
        # USB cameras from img files
        for cam in self.ext_image_keys:
            cam_path = os.path.join(path, cam)
            if os.path.exists(cam_path):
                # Get list of img files
                img_files = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
                images_info[cam] = {
                    'type': 'file_sequence',
                    'path': cam_path,
                    'files': img_files[:frame_num]  # Ensure we don't exceed frame count
                }
            else:
                images_info[cam] = None
        
        return {
            "state": state,
            "action": action,
            "images_info": images_info,
            "episode_len": frame_num,
            "episode_path": path,
        }

    def _get_decoded_video(self, episode_info, image_key: str, raw_bytes: bytes) -> np.ndarray:
        """Decodes video from raw H.264 bytes using AV, with caching."""
        cache_key = (episode_info, image_key)
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]

        frames = []
        if raw_bytes is None or len(raw_bytes) == 0:
            decoded_frames = np.array([])
        else:
            try:
                in_buffer = BytesIO(raw_bytes)
                container = av.open(in_buffer)
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
                decoded_frames = np.stack(frames) if frames else np.array([])
            except Exception as e:
                print(f"Warning: Failed to decode video (path {episode_info}, key {image_key}). Error: {e}")
                decoded_frames = np.array([])
        
        self._video_cache[cache_key] = decoded_frames
        return decoded_frames

    def _load_file_sequence(self, cam_info: Dict, start_idx: int, end_idx: int) -> np.ndarray:
        """Load a sequence of img images."""
        if cam_info is None or cam_info['type'] != 'file_sequence':
            return np.array([])
        
        frames = []
        cam_path = cam_info['path']
        files = cam_info['files']
        
        for i in range(start_idx, min(end_idx, len(files))):
            cache_key = (cam_path, files[i])
            
            if cache_key in self._image_cache:
                img_array = self._image_cache[cache_key]
            else:
                img_path = os.path.join(cam_path, files[i])
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    if img_array.ndim == 2:  # Grayscale to RGB
                        img_array = np.stack([img_array] * 3, axis=-1)
                    self._image_cache[cache_key] = img_array
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}. Error: {e}")
                    img_array = np.zeros((480, 640, 3), dtype=np.uint8)  # Default size
            
            frames.append(img_array)
        
        return np.stack(frames) if frames else np.array([])

    def parse_episode(self, episode_info):
        """
        Parses an episode to generate a training sample at a random timestep.
        """
        episode_data = self._extract_data_from_episode(episode_info)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        num_steps = episode_data["episode_len"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:  # case where robot doesn't move
            return False, None

        # Randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps)
        
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": episode_info.action
        }
        
        actions_full = episode_data["action"]
        target_qpos = actions_full[step_id : step_id + self.CHUNK_SIZE]
        
        # Parse state and action
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos

        if actions.shape[0] < self.CHUNK_SIZE:
            actions = np.pad(actions, ((0, self.CHUNK_SIZE - actions.shape[0]), (0, 0)), 'edge')

        state_dim = qpos.shape[1]
        state_indicator = np.ones(state_dim)

        # Parse images
        def parse_img(key):
            img_info = episode_data["images_info"].get(key)
            
            if key == 'head_camera':
                # Original video decoding for head camera
                if img_info is None:
                    return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
                
                video_frames = self._get_decoded_video(episode_info, key, img_info)
                
                if video_frames.ndim != 4:  # If decoding failed or empty
                    return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
                
                # Get image history
                start_idx = max(step_id - self.IMG_HISTORY_SIZE + 1, 0)
                imgs = video_frames[start_idx : step_id + 1]
            else:
                if img_info is None:
                    return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
                
                start_idx = max(step_id - self.IMG_HISTORY_SIZE + 1, 0)
                imgs = self._load_file_sequence(img_info, start_idx, step_id + 1)
                
                if imgs.ndim != 4:  # If loading failed
                    return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
            
            # Pad images if history is not full
            if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                pad_width = self.IMG_HISTORY_SIZE - imgs.shape[0]
                imgs = np.pad(imgs, ((pad_width, 0), (0,0), (0,0), (0,0)), 'edge')

            return imgs
        
        # Load all cameras
        cam_high = parse_img('head_camera')
        
        # Create masks
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISTORY_SIZE)
        cam_mask = np.array(
            [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
        )

        sample = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_mask,
        }
        
        # Add external cameras
        for key, name in zip(self.ext_image_keys, self.ext_image_names):
            sample[name] = parse_img(key)
            sample[name + '_mask'] = cam_mask
        
        return True, sample

    def parse_episode_state_only(self, episode_info):
        """
        Parses an episode to generate full state and action trajectories.
        """
        episode_data = self._extract_data_from_episode(episode_info)
        if not episode_data:
            return False, None
        
        qpos = episode_data["state"]
        actions = episode_data["action"]
        num_steps = episode_data["episode_len"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:
            return False, None

        state_traj = qpos[first_idx-1:]
        action_traj = actions[first_idx-1:]
        
        return True, {
            "state": state_traj,
            "action": action_traj
        }

if __name__ == "__main__":
    # --- Example Usage ---
    ds = BsonVLADataset()
    
    if len(ds) > 0:
        print(f"\n--- Testing get_item (state_only=False) for one item ---")
        sample = ds.get_item()
        print("Sample keys:", sample.keys())
        print("Meta:", sample['meta'])
        print("State shape:", sample['state'].shape)
        print("Actions shape:", sample['actions'].shape)
        print("Cam High shape:", sample['cam_high'].shape)
        for key, name in zip(ds.ext_image_keys, ds.ext_image_names):
            print(f"Cam {name} shape:", sample[name].shape)
        print("Cam masks:", sample['cam_high_mask'])

        print(f"\n--- Testing get_item (state_only=True) for one item ---")
        state_sample = ds.get_item(state_only=True)
        print("State sample keys:", state_sample.keys())
        print("Full state trajectory shape:", state_sample['state'].shape)
        print("Full action trajectory shape:", state_sample['action'].shape)
    else:
        print("\nDataset initialized but contains no valid episodes.")