#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XHand ZMQ Forwarder
Runs in the XHand environment and forwards hand control/observation data via ZMQ
"""

import argparse
import sys, os
import time
import yaml
import logging
import json

import numpy as np
import zmq

# XHand-specific imports will be added at runtime
# from xhand_tele_ops import XHandTeleOps


class XHandForwarder:
    """ZMQ forwarder for XHand communication"""
    
    def __init__(self, config_path, zmq_port=5557):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        logging.info(f"XHand forwarder listening on port {zmq_port}")
        
        # XHand configuration
        self.xhand_code_path = self.config['xhand_code_path']
        self.xhand_config = self.config['xhand_config']
        
        # Initialize XHand controller
        self._initialize_xhand()
    
    def _initialize_xhand(self):
        """Initialize the XHand controller"""
        os.chdir(self.xhand_code_path)
        from xhand_tele_ops import XHandTeleOps
        
        self.xhand_controller = XHandTeleOps(self.xhand_config)
        logging.info("XHand controller initialized successfully")
    
    def get_observations(self):
        """Get current hand state observations"""
        observations = {}
        
        # Get left hand data
        resp = self.xhand_controller.get_hand_full_info("hand_a", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            left_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['left_hand'] = left_hand_pos
        else:
            logging.error("Failed to get left hand data")
            observations['left_hand'] = [0.0] * 12
        
        # Get right hand data
        resp = self.xhand_controller.get_hand_full_info("hand_b", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            right_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['right_hand'] = right_hand_pos
        else:
            logging.error("Failed to get right hand data")
            observations['right_hand'] = [0.0] * 12
        
        return observations
    
    def execute_action(self, action_data):
        """Execute hand actions"""
        try:
            # Extract hand actions
            left_hand_action = action_data.get('left_hand', [0.0] * 12)
            right_hand_action = action_data.get('right_hand', [0.0] * 12)
            
            # Prepare data for xhand controller
            transform_data = {
                "left_hand": left_hand_action,
                "right_hand": right_hand_action
            }
            
            # Send actions to xhand
            logging.debug(f"Executing xHand action: L={left_hand_action}, R={right_hand_action}")
            self.xhand_controller.send_data_xhand(transform_data)
            
            return {'status': 'success'}
            
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            return {'error': str(e)}
    
    def handle_request(self, request):
        """Handle incoming ZMQ request"""
        command = request.get('command')
        
        if command == 'get_observations':
            return self.get_observations()
        
        elif command == 'execute_action':
            action_data = request.get('action_data')
            return self.execute_action(action_data)
        
        elif command == 'ping':
            return {'status': 'pong'}
        
        else:
            return {'error': f'Unknown command: {command}'}
    
    def run(self):
        """Main loop for handling requests"""
        logging.info("XHand forwarder started")
        
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
                    response = {'error': str(e)}
                
                # Send response
                self.socket.send_json(response)
                
        except KeyboardInterrupt:
            logging.info("Shutting down XHand forwarder...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Close ZMQ socket
        self.socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description="XHand ZMQ Forwarder")
    parser.add_argument("--config", type=str, default="deploy/mmk_xhand_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=5557,
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
    forwarder = XHandForwarder(args.config, args.port)
    forwarder.run()


if __name__ == "__main__":
    main()
