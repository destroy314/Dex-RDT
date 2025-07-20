#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMK Robot ZMQ Forwarder
Runs in the MMK environment and forwards robot control/observation data via ZMQ
"""

import argparse
import sys
import time
import yaml
import logging
import json
import base64

import numpy as np
import cv2
import zmq

# MMK-specific imports will be added at runtime
# from robots.airbots.airbot_mmk.airbot_com_mmk2_bson import AIRBOTMMK2
# from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2Config


class MMKForwarder:
    """ZMQ forwarder for MMK robot communication"""
    
    def __init__(self, config_path, zmq_port=5556):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        logging.info(f"MMK forwarder listening on port {zmq_port}")
        
        # Robot configuration
        self.ip = self.config["mmk_ip"]
        self.port = self.config["mmk_port"]
        self.mmk_code_path = self.config["mmk_code_path"]
        
        # Camera configuration - only handle internal camera
        self.internal_camera_name = "head_camera"
        
        # Component configuration
        self.components = ["left_arm", "right_arm", "head", "spine"]
        self.robot_cameras = {"head_camera": ["color"]}
        
        # Default action
        self.default_action = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0,
            0.15
        ]
        
        # Initialize robot only
        self._initialize_robot()
    
    def _initialize_robot(self):
        """Initialize the MMK robot"""
        sys.path.append(self.mmk_code_path)
        from robots.airbots.airbot_mmk.airbot_com_mmk2_bson import AIRBOTMMK2
        from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2Config
        
        config = AIRBOTMMK2Config(
            ip=self.ip,
            port=self.port,
            components=self.components,
            cameras=self.robot_cameras,
            default_action=self.default_action,
        )
        
        self.robot = AIRBOTMMK2(config=config)
        self.robot.reset(sleep_time=2)
        logging.info(f"Successfully connected to MMK robot at {self.ip}")
    
    def get_observations(self):
        """Get current robot state and internal camera observation"""
        # Get robot state
        robot_state_data = self.robot.get_low_dim_data()
        
        # Process joint positions
        left_joint_data = robot_state_data["/observation/left_arm/joint_state"]["data"]["pos"]
        right_joint_data = robot_state_data["/observation/right_arm/joint_state"]["data"]["pos"]
        
        qpos = np.array(left_joint_data + right_joint_data)
        
        # # Get internal camera image only
        # head_camera_image = self.robot._capture_images()[0][self.internal_camera_name]
        
        # # Encode image as base64 for transmission
        # _, buffer = cv2.imencode('.jpg', head_camera_image)
        # encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'qpos': qpos.tolist(),
            # 'head_camera_image': encoded_image
        }
    
    def execute_action(self, action):
        """Execute action on MMK robot"""
        # Execute actions
        self.robot.send_action(action + self.default_action[-3:])
        
        return {'status': 'success'}
    
    def reset(self):
        """Reset the robot"""
        self.robot.reset(sleep_time=2)
        return {'status': 'success'}
    
    def handle_request(self, request):
        """Handle incoming ZMQ request"""
        command = request.get('command')
        
        if command == 'get_observations':
            return self.get_observations()
        
        elif command == 'execute_action':
            action = request.get('action')
            return self.execute_action(action)
        
        elif command == 'reset':
            return self.reset()
        
        else:
            return {'error': f'Unknown command: {command}'}
    
    def run(self):
        """Main loop for handling requests"""
        logging.info("MMK forwarder started")
        
        try:
            while True:
                # Wait for request
                message = self.socket.recv_json()
                logging.debug(f"Received request: {message.get('command')}")
                
                # Process request
                try:
                    response = self.handle_request(message)
                except Exception as e:
                    logging.error(f"Error handling request: {e}")
                    import traceback
                    traceback.print_exc()
                    response = {'error': str(e)}
                
                # Send response
                self.socket.send_json(response)
                
        except KeyboardInterrupt:
            logging.info("Shutting down MMK forwarder...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Close ZMQ socket
        self.socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description="MMK Robot ZMQ Forwarder")
    parser.add_argument("--config", type=str, default="deploy/mmk_xhand_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=5556,
                        help="ZMQ port to listen on")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run forwarder
    forwarder = MMKForwarder(args.config, args.port)
    forwarder.run()


if __name__ == "__main__":
    main()
