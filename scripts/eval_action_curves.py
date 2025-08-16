#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script to compare ground truth actions with model predictions.
Loads a training episode, performs inference at regular intervals, and plots
GT action curves vs predicted action chunks for each axis.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import model and dataset utilities
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from data.bson_vla_dataset import BsonVLADataset

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_model(config, pretrained_path):
    """Initialize the RDT model from pretrained checkpoint"""
    logging.info("Creating RDT model...")
    
    # Create the vision encoder (Siglip)
    vision_encoder = SiglipVisionTower(vision_tower=config["vision_encoder_name"], args=None)
    image_processor = vision_encoder.image_processor
    
    # Load model from pretrained checkpoint
    logging.info(f"Loading pretrained model from {pretrained_path}")
    runner = RDTRunner.from_pretrained(pretrained_path)
    
    # Setup device
    runner = runner.to(device, dtype=dtype)
    vision_encoder = vision_encoder.to(device, dtype=dtype)
    
    # Set to evaluation mode
    runner.eval()
    vision_encoder.eval()
    
    return runner, vision_encoder, image_processor

def prepare_language_embeddings(lang_embeddings_path):
    """Prepare language embeddings for the model"""
    logging.info(f"Loading language embeddings from {lang_embeddings_path}")
    lang_dict = torch.load(lang_embeddings_path)
    logging.info(f"Using instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"].to(device, dtype=dtype)
    return lang_embeddings

def preprocess_images(images, image_processor, config):
    """Preprocess images for model input"""
    # Background image for padding
    background_color = np.array([
        int(x*255) for x in image_processor.image_mean
    ], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((
        image_processor.size["height"], 
        image_processor.size["width"], 3), dtype=np.uint8
    ) * background_color
    
    # Preprocess images
    image_tensor_list = []
    for image in images:
        if image is None:
            image = Image.fromarray(background_image)
        else:
            image = Image.fromarray(image)
        
        if config.get("image_aspect_ratio", "pad") == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_list.append(image)
    
    image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
    return image_tensor

def run_inference(policy, vision_encoder, image_processor, config, lang_embeddings, 
                 images, qpos):
    """Run model inference with given observations"""
    # Preprocess images
    image_tensor = preprocess_images(images, image_processor, config)
    
    # Process with vision encoder
    image_embeds = vision_encoder(image_tensor).detach()
    image_embeds = image_embeds.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)
    
    # Prepare proprioception
    proprio = torch.tensor(qpos, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    
    # Setup action mask
    action_mask = torch.ones(1, 1, config['state_dim'], device=device, dtype=dtype)
    
    # Setup control frequency
    ctrl_freq = torch.tensor([config['control_frequency']], device=device)
    
    # Run inference
    with torch.no_grad():
        actions = policy.predict_action(
            lang_tokens=lang_embeddings,
            lang_attn_mask=torch.ones(
                lang_embeddings.shape[:2], dtype=torch.bool,
                device=lang_embeddings.device),
            img_tokens=image_embeds,
            state_tokens=proprio,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freq
        )
    
    return actions.squeeze(0).to(torch.float).cpu().numpy()

def load_episode_data(dataset, episode_idx):
    """Load full episode data including images"""
    # Get episode with full trajectories
    success, episode_data = dataset.parse_episode_state_only(dataset.episode_infos[episode_idx])
    if not success:
        raise ValueError(f"Failed to load episode {episode_idx}")
    
    # Extract episode info for image loading
    episode_info = dataset.episode_infos[episode_idx]
    episode_raw_data = dataset._extract_data_from_episode(episode_info)
    
    # Load images for each timestep
    images_by_timestep = []
    num_steps = len(episode_data['state'])
    
    for step in range(num_steps):
        step_images = []
        
        # Load images for all cameras at this timestep
        for camera_name in dataset.image_keys:
            if camera_name == 'head_camera':
                # Handle head camera (video stream)
                if episode_raw_data['images_info']['head_camera'] is not None:
                    video_data = episode_raw_data['images_info']['head_camera']
                    if step < len(video_data):
                        # Decode video frame (simplified - would need proper video decoding)
                        # For now, use a placeholder
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Handle external cameras (file sequences)
                cam_info = episode_raw_data['images_info'].get(camera_name)
                if cam_info and cam_info['type'] == 'file_sequence':
                    if step < len(cam_info['files']):
                        img_path = os.path.join(cam_info['path'], cam_info['files'][step])
                        if os.path.exists(img_path):
                            img = np.array(Image.open(img_path))
                        else:
                            img = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            step_images.append(img)
        
        images_by_timestep.append(step_images)
    
    return episode_data, images_by_timestep

def evaluate_episode(args):
    """Main evaluation function"""
    logging.basicConfig(level=logging.INFO)
    
    # Hardcoded configuration (from deploy/mmk_xhand_config.yaml)
    config = {
        'camera_names': ['head_camera', 'cam_left_wrist', 'cam_third_view', 'cam_right_wrist'],
        'ext_cam_ids': [0, 2, 6],
        'img_history_size': 1,
        'image_aspect_ratio': 'pad',
        'seed': 42,
        'state_dim': 36,  # 6 DoF for each arm + 12 DoF for each hand
        'chunk_size': 32,  # Action chunk size for inference
        'vision_encoder_name': 'google/siglip-so400m-patch14-384',
        'control_frequency': 20.0,
        'use_actions_interpolation': False,
        'max_steps': 100000
    }
    
    # Set random seed
    set_seed(config["seed"])
    
    # Create model
    policy, vision_encoder, image_processor = create_model(config, args.pretrained_model_path)
    
    # Load language embeddings
    lang_embeddings = prepare_language_embeddings(args.lang_embeddings_path)
    
    # Create dataset
    dataset = BsonVLADataset(bson_dir=args.data_dir)
    
    if len(dataset) == 0:
        raise ValueError("No episodes found in dataset")
    
    # Load episode data
    episode_idx = args.episode_idx if args.episode_idx < len(dataset) else 0
    logging.info(f"Loading episode {episode_idx} out of {len(dataset)} episodes")
    
    episode_data, images_by_timestep = load_episode_data(dataset, episode_idx)
    
    gt_states = episode_data['state']
    gt_actions = episode_data['action']
    num_steps = len(gt_states)
    
    logging.info(f"Episode has {num_steps} steps")
    logging.info(f"State shape: {gt_states.shape}, Action shape: {gt_actions.shape}")
    
    # Run inference at regular intervals
    inference_interval = args.inference_interval
    inference_steps = list(range(0, num_steps, inference_interval))
    
    predicted_actions = []
    inference_timesteps = []
    
    logging.info(f"Running inference at {len(inference_steps)} timesteps...")
    
    for step in inference_steps:
        if step >= num_steps:
            break
            
        logging.info(f"Running inference at step {step}")
        
        # Get current state and images
        current_state = gt_states[step]
        current_images = images_by_timestep[step]
        
        # Run inference
        pred_action_chunk = run_inference(
            policy, vision_encoder, image_processor, config, lang_embeddings,
            current_images, current_state
        )
        
        predicted_actions.append(pred_action_chunk)
        inference_timesteps.append(step)
    
    # Plot results
    plot_action_curves(gt_actions, predicted_actions, inference_timesteps, 
                      config, args.output_dir, episode_idx)

def plot_action_curves(gt_actions, predicted_actions, inference_timesteps, 
                      config, output_dir, episode_idx):
    """Plot GT vs predicted action curves for each axis"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Action dimension names
    action_names = []
    # Left arm (6 DoF)
    for i in range(6):
        action_names.append(f"left_arm_joint_{i+1}")
    # Left hand (12 DoF)
    for i in range(12):
        action_names.append(f"left_hand_joint_{i+1}")
    # Right arm (6 DoF)
    for i in range(6):
        action_names.append(f"right_arm_joint_{i+1}")
    # Right hand (12 DoF)
    for i in range(12):
        action_names.append(f"right_hand_joint_{i+1}")
    
    num_axes = config['state_dim']
    chunk_size = config['chunk_size']
    
    # Plot each axis
    for axis in range(num_axes):
        plt.figure(figsize=(12, 6))
        
        # Plot GT trajectory
        timesteps = np.arange(len(gt_actions))
        plt.plot(timesteps, gt_actions[:, axis], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # Plot predicted action chunks
        colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_actions)))
        
        for i, (pred_chunk, inference_step) in enumerate(zip(predicted_actions, inference_timesteps)):
            # Plot the predicted chunk starting from the inference timestep
            chunk_timesteps = np.arange(inference_step, min(inference_step + chunk_size, len(gt_actions)))
            chunk_actions = pred_chunk[:len(chunk_timesteps), axis]
            
            plt.plot(chunk_timesteps, chunk_actions, '--', color=colors[i], 
                    linewidth=1.5, alpha=0.7, label=f'Prediction @step {inference_step}')
        
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.title(f'Episode {episode_idx} - {action_names[axis]} (Axis {axis})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'episode_{episode_idx}_axis_{axis}_{action_names[axis]}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved plot for axis {axis} ({action_names[axis]}) to {output_path}")
    
    # Create a summary plot with all axes
    fig, axes = plt.subplots(6, 6, figsize=(20, 16))
    axes = axes.flatten()
    
    for axis in range(min(num_axes, 36)):  # Limit to 36 subplots
        ax = axes[axis]
        
        # Plot GT
        timesteps = np.arange(len(gt_actions))
        ax.plot(timesteps, gt_actions[:, axis], 'b-', linewidth=1, label='GT', alpha=0.8)
        
        # Plot predictions
        for i, (pred_chunk, inference_step) in enumerate(zip(predicted_actions, inference_timesteps)):
            chunk_timesteps = np.arange(inference_step, min(inference_step + chunk_size, len(gt_actions)))
            chunk_actions = pred_chunk[:len(chunk_timesteps), axis]
            ax.plot(chunk_timesteps, chunk_actions, '--', alpha=0.6, linewidth=0.8)
        
        ax.set_title(f'{action_names[axis]}', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for axis in range(num_axes, 36):
        axes[axis].set_visible(False)
    
    plt.suptitle(f'Episode {episode_idx} - All Action Axes Comparison', fontsize=14)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, f'episode_{episode_idx}_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved summary plot to {summary_path}")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate RDT model action predictions")
    
    # Model configuration (config is now hardcoded in the script)
    parser.add_argument("--pretrained-model-path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--lang-embeddings-path", type=str, required=True,
                        help="Path to pre-computed language embeddings")
    
    # Dataset configuration
    parser.add_argument("--data-dir", type=str, default="data/ours",
                        help="Path to dataset directory")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="Episode index to evaluate")
    
    # Evaluation configuration
    parser.add_argument("--inference-interval", type=int, default=10,
                        help="Interval between inference steps")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for plots")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = get_args()
    evaluate_episode(args)

if __name__ == "__main__":
    main()
