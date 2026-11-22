import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import cv2
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import logging
from urdf_models import models_data

'''
🐞 (Ladybug): For debugging.
ℹ️ (Information): For informational messages.
🚨 (Alert): For errors or warnings.
✅ (Checkmark): For completion or success.
⏱️ (Timer Clock): For timing or performance metrics.
🟡 (Warning)
'''

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Define the format of log messages
    # NOTE: To save logs to a file, uncomment the following line
    # handlers=[
    #     logging.FileHandler("app.log"),  # Save logs to a file
    #     logging.StreamHandler()  # Also print logs to console
    # ]
)

class RoboticPushingEnv(Env):
    """
    A PyBullet-based environment for training a robotic arm to push objects to a target area
    using multimodal observations (vision and proprioception).
    """
    
    def __init__(self, 
                 render_mode=None, 
                 max_steps=1000, 
                 object_spawn_range_x=(0.5, 0.6), 
                 object_spawn_range_y=(-0.1, 0.1),
                 observation_type="multimodal"):
        """
        Initialize the robotic pushing environment.

        Args:
            render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None)
            max_steps (int): Maximum number of steps per episode
            object_spawn_range_x (tuple): Range for spawning objects along x-axis
            object_spawn_range_y (tuple): Range for spawning objects along y-axis
            observation_type (str): Type of observation to use ("multimodal", "multimodal_no_goal", "vision_only", "proprio_only")
        """
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_counter = 0
        
        # Connect to PyBullet
        self.physics_client_id = self._connect_physics_client()
        
        # Set simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # 0: stepped mode, 1: real-time mode
        
        # Initialize environment elements
        self.plane_id = None
        self.table_id = None
        self.robot_id = None
        self.object_id = None
        self.target_id = None
        
        # Position parameters
        self.table_position = [0.5, 0, 0]
        self.robot_position = [0, 0, 0.6]
        self.object_start_position = [0.6, 0.0, 0.65]
        self.object_start_orientation_yaw = np.pi/2
        self.target_position = [0.7, 0.0, 0.625]

        # Table dimensions (estimated from the scene)
        self.table_width = 1.0
        self.table_height = 0.6
        self.table_length = 1.5
        
        # Camera parameters
        self.camera_position = [1.2, 0.0, 1.2]
        self.camera_target = [0.7, 0, 0.8]
        self.camera_up_vector = [0, 0, 1]
        self.camera_fov = 60
        self.camera_aspect = 1.0
        self.camera_near_val = 0.01
        self.camera_far_val = 2.0
        self.camera_width = 224
        self.camera_height = 224
        
        # Robot control parameters
        self.end_effector_index = 11  # Using 'panda_grasptarget_hand' as the pushing point
        self.num_joints = 0
        self.joint_limits = [
            (-2.9671, 2.9671),  # panda_joint1
            (-1.8326, 1.8326),  # panda_joint2
            (-2.9671, 2.9671),  # panda_joint3
            (-3.1416, 0.0),     # panda_joint4
            (-2.9671, 2.9671),  # panda_joint5
            (-0.0873, 3.8223),  # panda_joint6
            (-2.9671, 2.9671),  # panda_joint7
        ]
        self.target_joint_positions = None

        # Save the user-defined spawn ranges
        self.object_spawn_range_x = object_spawn_range_x
        self.object_spawn_range_y = object_spawn_range_y
        
        # Create view and projection matrices for the camera
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up_vector
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near_val,
            farVal=self.camera_far_val
        )

        # Load URDF models
        self.models = models_data.model_lib()
        self.ycb_object = "pudding_box" # Available objects: ['cracker_box', 'sugar_box', "soap", "pudding_box", "gelatin_box", "blue_tea_box"]
        
        # Initialize the environment
        self._load_environment()

        # Calculate and store the target position 2D
        _, seg_img = self._get_camera_image()
        self.target_position_2d = self._get_target_position_2d(seg_img)

        # Define observation space based on the chosen observation type
        self.observation_type = observation_type
        if self.observation_type == "vision_only":
            self.observation_space = spaces.Dict({
                # Normalized bbox
                'vision': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                # Target x,y
                'target': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            })
        elif self.observation_type == "proprio_only":
            self.observation_space = spaces.Dict({
                # End-effector pos & vel
                'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            })
        elif self.observation_type == "multimodal_no_goal":
            self.observation_space = spaces.Dict({
                # Normalized bbox
                'vision': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                # End-effector pos (3) & vel (3)
                'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            })
        elif self.observation_type == "multimodal":
            self.observation_space = spaces.Dict({
                # Normalized bbox
                'vision': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                # Target x,y
                'target': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                # End-effector pos (3) & vel (3)
                'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
        
        # Define action space (end-effector movement in x, y directions)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # User parameters
        self.distance_threshold = 0.03  # Distance threshold for object-target proximity    
    
    def _connect_physics_client(self):
        """Connect to the PyBullet physics server."""
        if self.render_mode == "human":
            # return p.connect(p.GUI)
            # Suppress PyBullet GUI output
            with open(os.devnull, 'w') as devnull:
                original_stdout = os.dup(1)  # Duplicate stdout
                os.dup2(devnull.fileno(), 1)  # Redirect stdout to devnull
                client = p.connect(p.GUI)
                os.dup2(original_stdout, 1)  # Restore original stdout
            return client
        
        else:
            return p.connect(p.DIRECT)
    
    def _load_environment(self):
        """Load all objects in the environment."""
        # Reset simulation
        p.resetSimulation()
        
        # Load plane (ground)
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        
        # Load table
        self.table_id = p.loadURDF("table/table.urdf", basePosition=self.table_position)
        
        # Load robot arm (Panda)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=self.robot_position, 
                                   useFixedBase=True)
        
        # Get number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Set initial robot configuration
        self._reset_robot()
        
        # Load pushable object
        # self.object_id = p.loadURDF("cube_small.urdf", basePosition=self.object_start_position)
        # Load a YCB object
        available_objects = ['cracker_box', 'sugar_box', "soap", "pudding_box", "gelatin_box", "blue_tea_box"]
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.object_id = p.loadURDF(self.models[self.ycb_object], basePosition=self.object_start_position, baseOrientation=p.getQuaternionFromEuler([0, 0, self.object_start_orientation_yaw]), flags=flags)
        
        # Create target visual (green area)
        target_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.05, 0.05, 0.005], 
            rgbaColor=[0, 1, 0, 0.6]
        )
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=target_visual, 
            basePosition=self.target_position
        )

        p.setGravity(0, 0, -9.81)
        
        # Add camera visualization
        self._add_camera_visualization()
    
    def _reset_robot(self):
        """Reset robot to home position."""
        # Initial joint positions for home position - configured for pushing task
        # These values position the end effector above the table at a good height for pushing
        # initial_joint_positions = [0, -0.3, 0, -2.2, 0, 2.0, 0.8]
        # initial_joint_positions = [0.0, 0.347, -0.0, -2.431, -0.0, 2.732, 0.625] # robot pose 1
        # initial_joint_positions = [0.0, 0.347, -0.0, -2.431, -0.0, 2.85, 0.78] # robot pose 2
        initial_joint_positions = [0.0, 0.087, 0.0, -3.068, 0.0, 3.227, 0.78] # robot pose, no collision with object
        
        # Get current number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Set joint positions for the 7 controllable joints
        for i in range(7):
            p.resetJointState(self.robot_id, i, initial_joint_positions[i])
            
            # Set position control with appropriate force limits from joint info
            max_force = 87.0 if i < 4 else 12.0  # First 4 joints have higher force limits
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=initial_joint_positions[i],
                force=max_force
            )
        
        # Set gripper finger joints to close position
        p.resetJointState(self.robot_id, 9, 0.0)  # Left finger closed 
        p.resetJointState(self.robot_id, 10, 0.0)  # Right finger closed
        
        # Control gripper
        for i in [9, 10]:  # Finger joints
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, # Fingers closed
                force=20.0
            )
    
    def _add_camera_visualization(self):
        """Add visual elements to represent the camera in the simulation."""
        # Create a small red sphere to represent camera position
        camera_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.03, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.camera_body = p.createMultiBody(
            baseVisualShapeIndex=camera_visual, 
            basePosition=self.camera_position
        )
        
        # Create a line from camera to target
        p.addUserDebugLine(
            self.camera_position,
            self.camera_target,
            lineColorRGB=[1, 0, 0],
            lineWidth=2.0
        )
    
    def _get_camera_image(self):
        """Capture an image from the camera."""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )
        
        return rgb_img, seg_img
    
    def _get_object_bounding_box(self, rgb_img, seg_img):
        """
        Get the bounding box of the pushable object from segmentation image.
        
        Returns:
            bbox (np.ndarray): [x, y, width, height] normalized to [0, 1]
        """
        # Get object segmentation mask
        
        # Convert segmentation image from rgba to single channel
        seg_img_single = seg_img & ((1 << 24) - 1)  # Extract object ID from the seg_img
        
        # Create mask where the object ID matches
        object_mask = np.zeros_like(seg_img_single, dtype=np.uint8)
        object_mask[seg_img_single == self.object_id] = 255
        
        # Debug visualization if needed
        cv2.imwrite('debug_mask.png', object_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # Get bounding box from the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Normalize to [0, 1]
            x_norm = x / self.camera_width
            y_norm = y / self.camera_height
            w_norm = w / self.camera_width
            h_norm = h / self.camera_height

             # Store this as the last known valid detection
            self.last_detected_bbox = np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32)
            self.frames_since_detection = 0
            
            # logging.info(f"✅ [vision] Found object with bbox")
            return self.last_detected_bbox, True
        
        # If detection method fails and we have a previous detection, use that
        if hasattr(self, 'last_detected_bbox'):
            self.frames_since_detection += 1
            logging.info(f"🟡 [vision] Object occluded, using tracking. Frames since detection: {self.frames_since_detection}")
            return self.last_detected_bbox, False

        # Only if we have no previous detection and all methods fail
        logging.info(f"❌ [vision] Object not found and no tracking data available")
        # Initialize a reasonable default based on object start position
        # This is better than completely arbitrary values
        default_bbox = np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32)
        self.last_detected_bbox = default_bbox
        self.frames_since_detection = 0
        return default_bbox, False
    
    def _get_target_position_2d(self, seg_img):
        """
        Get the 2D position of the target area in the table frame using the segmentation image.
        
        Args:
            seg_img (np.ndarray): Segmentation image from the camera
        
        Returns:
            np.ndarray: Normalized target [x, y] position
        """
        # Convert segmentation image from rgba to single channel
        seg_img_single = seg_img & ((1 << 24) - 1)  # Extract object ID from the seg_img
        
        # Create mask where the target ID matches
        target_mask = np.zeros_like(seg_img_single, dtype=np.uint8)
        target_mask[seg_img_single == self.target_id] = 255

        # Debug visualization if needed
        cv2.imwrite('debug_target_mask.png', target_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # Get bounding box from the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate the center of the bounding box
            target_x = x + w / 2
            target_y = y + h / 2
            
            # Change to [0,1] range to match object bbox
            target_x_norm = target_x / self.camera_width
            target_y_norm = target_y / self.camera_height
            
            return np.array([target_x_norm, target_y_norm], dtype=np.float32)
        
        # If target not found, return default position
        # TODO: either exit the code and implement something else because the target should be visible once only at the start
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def _get_proprioceptive_data(self):
        """
        Get proprioceptive data: end-effector position and velocity.
        
        Returns:
            np.ndarray: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        """
        # Get end-effector state
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index, computeLinkVelocity=1)
        
        # Extract position and velocity
        position = end_effector_state[0]  # Position in world frame
        velocity = end_effector_state[6]  # Linear velocity in world frame
        
        return np.array([*position, *velocity], dtype=np.float32)
    
    def _get_observation(self):
        """
        Get the multimodal observation from the environment.
        
        Returns:
            dict: Dictionary with vision, target, and proprioception data
        """
        # Get camera image
        rgb_img, seg_img = self._get_camera_image()
        
        # Extract observations
        object_bbox, is_direct_detection = self._get_object_bounding_box(rgb_img, seg_img)
        proprio_data = self._get_proprioceptive_data()

        # object_bbox = self._get_object_bounding_box(rgb_img, seg_img)
        # occlusion_detected = self._detect_occlusion(object_bbox, rgb_img, seg_img)
        self.current_occlusion = not is_direct_detection  # Store for reward calculation

        # Return different observation components based on observation type
        if self.observation_type == "vision_only":
            # logging.info(f"ℹ️ [observation] Vision only observation")
            return {
                'vision': object_bbox,
                'target': self.target_position_2d
            }
        elif self.observation_type == "proprio_only":
            # logging.info(f"ℹ️ [observation] Proprioception only observation")
            return {
                'proprioception': proprio_data
            }
        elif self.observation_type == "multimodal_no_goal":
            # logging.info(f"ℹ️ [observation] Multimodal observation without goal")
            return {
                'vision': object_bbox,
                'proprioception': proprio_data
            }
        elif self.observation_type == "multimodal":
            # logging.info(f"ℹ️ [observation] Multimodal observation with goal")
            return {
                'vision': object_bbox,
                'target': self.target_position_2d,
                'proprioception': proprio_data
            }
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
        
        # return {
        #     'vision': object_bbox,
        #     'target': self.target_position_2d,
        #     'proprioception': proprio_data
        # }
    
    def _is_object_at_target(self):
        """
        Check if the pushable object is at the target position.
        
        Returns:
            bool: True if object is at target, False otherwise
        """

        # Get latest detection from vision system
        if not hasattr(self, 'last_detected_bbox'):
            return False
        # Extract center of object bounding box
        bbox = self.last_detected_bbox
        self.obj_center_x = bbox[0] + bbox[2]/2
        self.obj_center_y = bbox[1] + bbox[3]/2
        
        # Calculate distance to target in normalized coordinates
        dist = np.sqrt((self.obj_center_x - self.target_position_2d[0])**2 + 
                    (self.obj_center_y - self.target_position_2d[1])**2)
        
        # Check if distance is within threshold
        return dist < self.distance_threshold
    
    def _calculate_reward(self):
        """Simplified reward function for the basic pushing task."""

        # Weights for each reward component
        time_penalty_weight = 0.05       # Reduced to be less punitive
        goal_reward_weight = 5.0         # Goal completion is important
        distance_reward_weight = 2.0      
        progress_reward_weight = 2.0     # Increased significantly - progress is crucial
        approach_reward_weight = 1.0     
        movement_reward_weight = 1.5     # Moving the object is important
        occlusion_penalty_weight = 0.2   # Discourage blocking the object view

        # 1. Time penalty
        time_penalty = -0.01

        # 2. Big positive reward for reaching target
        goal_reward = 50.0 if self._is_object_at_target() else 0.0
        
        # 3. Distance-based reward (inversely proportional)
        # Calculate distance to target in the xy-plane
        dist = np.sqrt((self.obj_center_x - self.target_position_2d[0])**2 +
                       (self.obj_center_y - self.target_position_2d[1])**2)
        # Smoother distance reward using exponential function
        distance_reward = -1.0 * (1.0 - np.exp(-4.0 * dist))
        
        # 4. Progress reward (for reducing distance)
        if hasattr(self, 'prev_dist'):
            progress_reward = max(0, self.prev_dist - dist) * 2.0
            self.prev_dist = dist
        else:
            progress_reward = 0
            self.prev_dist = dist
        
        # 5. End-effector to object proximity reward
        ee_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        # More generous approach reward that doesn't diminish as quickly
        ee_to_object_dist = np.sqrt((ee_pos[0] - self.obj_center_x)**2 + 
                            (ee_pos[1] - self.obj_center_y)**2)
        approach_reward = 1.0 / (1.0 + 3.0 * ee_to_object_dist)  # Much stronger signal

        # 6. Object movement reward
        if hasattr(self, 'prev_object_pos'):
            object_movement = np.sqrt((self.obj_center_x - self.prev_object_pos[0])**2 +
                                    (self.obj_center_y - self.prev_object_pos[1])**2)

            # Only meaningful movement gets rewarded
            if object_movement > 0.003:  # Small threshold for meaningful movement
                movement_reward = min(5.0, object_movement * 10.0)

                # Directional reward (higher for moving toward target)
                if hasattr(self, 'prev_dist') and self.prev_dist > dist:
                    movement_reward *= 2.0  # Double reward for correct direction
            else:
                movement_reward = 0.0
        else:
            movement_reward = 0.0

        self.prev_object_pos = (self.obj_center_x, self.obj_center_y)

        # 7. Add occlusion penalty
        occlusion_penalty = -5.0 if self.current_occlusion else 0.0
        
        # Total reward
        reward = (time_penalty_weight * time_penalty +
                goal_reward_weight * goal_reward +
                distance_reward_weight * distance_reward +
                progress_reward_weight * progress_reward +
                approach_reward_weight * approach_reward +
                movement_reward_weight * movement_reward +
                occlusion_penalty_weight * occlusion_penalty)
        
        return reward
    
    def _apply_action(self, action):
        """
        Apply action to control the robot arm.
        
        Args:
            action (np.ndarray): [delta_x, delta_y] for end-effector movement
        """
        
        # Add action smoothing to prevent jerky movements
        if hasattr(self, 'previous_action'):
            # Blend with previous action (70% current, 30% previous)
            action = 0.7 * action + 0.3 * self.previous_action
        
        # Store current action for next step
        self.previous_action = action.copy()
        
        # Scale action to appropriate range (e.g., cm)
        action_scale = 0.02  # 2 cm per step when action = 1.0
        delta_x = action[0] * action_scale
        delta_y = action[1] * action_scale
        
        # Get current end-effector position
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        current_position = end_effector_state[0]
        current_orientation = end_effector_state[1]
        
        # Calculate target position (only modify x, y)
        target_position_for_ik = [
            current_position[0] + delta_x,
            current_position[1] + delta_y,
            current_position[2]  # Keep z-coordinate the same
        ]

        # Safety check
        # if not self._is_pose_safe(target_position_for_ik, current_orientation):
        #     # Try smaller movement if original is unsafe
        #     scale_factor = 0.5
        #     safe_target = None
            
        #     # Try progressively smaller actions until finding a safe one
        #     while scale_factor >= 0.1:
        #         adjusted_position = [
        #             current_position[0] + (delta_x * scale_factor),
        #             current_position[1] + (delta_y * scale_factor),
        #             current_position[2]
        #         ]
                
        #         if self._is_pose_safe(adjusted_position, current_orientation):
        #             safe_target = adjusted_position
        #             logging.info(f"🟡 [safety] Reduced action to {scale_factor*100}% for safety")
        #             break
                    
        #         scale_factor *= 0.5
            
        #     if safe_target is None:
        #         logging.info("🚨 [safety] No safe action found, skipping")
        #         return
            
        #     target_position_for_ik = safe_target
        
        # Use inverse kinematics to get joint positions for target end-effector position
        # We maintain the current orientation for stable pushing
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position_for_ik,
            current_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Apply joint positions (only to the first 7 joints of the Panda arm)
        for i in range(7):
            # Get appropriate force limits
            max_force = 87.0 if i < 4 else 12.0  # Based on the joint info
            
            # Clip to joint limits
            lower_limit, upper_limit = self.joint_limits[i]
            target_position_for_ik = min(
                max(joint_positions[i], lower_limit), upper_limit)
            
            # Set position control
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position_for_ik,
                force=max_force,
                maxVelocity=2.0  # Limit velocity for smoother movement
            )
        
        # Step the simulation multiple times to let the arm move
        for _ in range(10):
            p.stepSimulation()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation (dict): Initial observation
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset simulation
        self._load_environment()
        
        # Randomize object position (optional)
        # if options and 'randomize' in options and options['randomize']:
        # x = np.random.uniform(0.5, 0.6)
        # y = np.random.uniform(-0.1, 0.1)
        # self.object_start_position = [x, y, 0.65]
        
        x = np.random.uniform(*self.object_spawn_range_x)
        y = np.random.uniform(*self.object_spawn_range_y)
        self.object_start_position = [x, y, 0.65]

        self.object_orientation = p.getQuaternionFromEuler([0, 0, self.object_start_orientation_yaw])
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self.object_id,
            posObj=self.object_start_position,
            ornObj=self.object_orientation
        )
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): [delta_x, delta_y] for end-effector movement
        
        Returns:
            observation (dict): Current observation
            reward (float): Reward for this step
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated (e.g., due to max_steps)
            info (dict): Additional information
        """
        # Apply action
        self._apply_action(action)
        
        # Increment step counter
        self.step_counter += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        object_at_target = self._is_object_at_target()
        # Extra reward for quick completion
        if object_at_target:
            time_bonus = max(0, self.max_steps - self.step_counter) * 0.1
            reward += time_bonus

        object_off_table = self._is_object_off_table()

        # Add a large negative reward if object falls off table
        if object_off_table:
            reward -= 20.0  # Much larger penalty for failure

        terminated = object_at_target or object_off_table
        truncated = self.step_counter >= self.max_steps
        
        # Additional info
        info = {
            'is_success': object_at_target,
            'step_count': self.step_counter,
            'termination_reason': 'goal_reached' if object_at_target else 
                                ('object_off_table' if object_off_table else 'not_terminated')
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            np.ndarray or None: RGB image if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            rgb_img, _, _ = self._get_camera_image()
            return rgb_img
        
        return None
    
    def close(self):
        """Close the environment and disconnect from PyBullet."""
        p.disconnect(self.physics_client_id)
    
    def _is_pose_safe(self, target_position, target_orientation=None):
        """
        Check if a given end-effector pose is safe (no collisions with table or self).
        
        Args:
            target_position (list): Target position [x, y, z]
            target_orientation (list, optional): Target orientation as quaternion
            
        Returns:
            bool: True if the pose is safe, False otherwise
        """
        # Get current end-effector orientation if not provided
        if target_orientation is None:
            current_orientation = p.getLinkState(self.robot_id, self.end_effector_index)[1]
            target_orientation = current_orientation
        
        # 1. Check table height bounds
        min_safe_height = self.table_height + 0.02  # Minimum safe height above table
        
        if target_position[2] < min_safe_height:
            logging.info("🚨 [safety] Target position too low, would collide with table")
            return False
        
        # 2. Check table bounds (assuming the table is rectangular)
        table_half_length = self.table_length/2  # Adjust based on your table dimensions
        table_half_width = self.table_width/2   # Adjust based on your table dimensions
        
        table_min_x = self.table_position[0] - table_half_length
        table_max_x = self.table_position[0] + table_half_length
        table_min_y = self.table_position[1] - table_half_width
        table_max_y = self.table_position[1] + table_half_width
        
        # Check if target is far outside table bounds (too far is also unsafe)
        max_reach = 0.9  # Maximum safe reach of the robot arm

        if (target_position[0] < table_min_x - 0.1 or 
            target_position[0] > table_max_x + 0.1 or
            target_position[1] < table_min_y - 0.1 or
            target_position[1] > table_max_y + 0.1 or
            np.linalg.norm(np.array(target_position)) > max_reach):
            logging.info("🚨 [safety] Target position outside safe workspace")
            return False
        
        # 3. Check if IK solution exists and is feasible
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Save current joint positions
        current_joints = []
        for i in range(7):  # 7 DOF for Panda arm
            current_joints.append(p.getJointState(self.robot_id, i)[0])
        
        # Check if IK solution is within joint limits
        for i in range(7):
            lower_limit, upper_limit = self.joint_limits[i]
            if joint_positions[i] < lower_limit or joint_positions[i] > upper_limit:
                logging.info(f"🚨 [safety] Joint {i} would exceed limits")
                return False
        
        # 4. Use PyBullet's built-in collision detection
        # Temporarily move to target position
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_positions[i])
        
        # Check for collisions
        collision = False
        if p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id):
            logging.info("🚨 [safety] Would collide with table at target position")
            collision = True
        
        # Also check for self-collisions
        for i in range(p.getNumJoints(self.robot_id)):
            for j in range(i+1, p.getNumJoints(self.robot_id)):
                if p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, linkIndexA=i, linkIndexB=j):
                    logging.info(f"🚨 [safety] Self-collision detected between links {i} and {j}")
                    collision = True
                    break
            if collision:
                break
        
        # Restore original joint positions
        for i in range(7):
            p.resetJointState(self.robot_id, i, current_joints[i])
        
        return not collision
    
    def _is_object_off_table(self):
        """
        Check if the pushable object has fallen off the table.
        
        Returns:
            bool: True if object is off the table, False otherwise
        """
        # Get object position
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Check z-position (if it's below table height)
        if object_pos[2] < self.table_position[2] + 0.05:  # Add small buffer over table height
            logging.info("🚨 [termination] Object has fallen below table height")
            return True
        
        # Calculate table bounds
        table_half_length = self.table_length/2
        table_half_width = self.table_width/2
        
        table_min_x = self.table_position[0] - table_half_length
        table_max_x = self.table_position[0] + table_half_length
        table_min_y = self.table_position[1] - table_half_width
        table_max_y = self.table_position[1] + table_half_width
        
        # Check if object is outside table bounds in x-y plane
        if (object_pos[0] < table_min_x or object_pos[0] > table_max_x or
            object_pos[1] < table_min_y or object_pos[1] > table_max_y):
            logging.info("🚨 [termination] Object has moved outside table boundaries")
            return True
        
        return False
    
    def _detect_occlusion(self, object_bbox, rgb_img, seg_img):
        """Detect if the robot is occluding the object from camera view."""
        
        # Method 1: Check if bounding box area suddenly decreased
        if hasattr(self, 'prev_bbox'):
            prev_area = self.prev_bbox[2] * self.prev_bbox[3]
            current_area = object_bbox[2] * object_bbox[3]
            
            # If area reduced significantly and object is not near edges
            if current_area < prev_area * 0.7:
                # Get robot segmentation mask
                seg_img_single = seg_img & ((1 << 24) - 1) 
                robot_mask = np.zeros_like(seg_img_single, dtype=np.uint8)
                
                # Check for robot pixels in the expected object region
                for i in range(1, self.num_joints + 1):  # +1 for base
                    robot_mask[seg_img_single == self.robot_id + i - 1] = 255
                
                # Approximate where object should be based on previous detection
                prev_x = int(self.prev_bbox[0] * self.camera_width)
                prev_y = int(self.prev_bbox[1] * self.camera_height)
                prev_w = int(self.prev_bbox[2] * self.camera_width)
                prev_h = int(self.prev_bbox[3] * self.camera_height)
                
                # Check if robot is in the expected object region
                expected_object_region = robot_mask[prev_y:prev_y+prev_h, prev_x:prev_x+prev_w]
                if np.sum(expected_object_region) > 0:
                    return True
        
        # Save current bbox for next comparison
        self.prev_bbox = object_bbox.copy()
        return False

# Example usage
if __name__ == "__main__":

    # Create environment
    env = RoboticPushingEnv(render_mode="human", max_steps=100000)
    
    # Reset environment
    logging.info("ℹ️ Resetting environment...")
    obs, _ = env.reset()
    
    # Run a few random steps
    for _ in range(1000):
        action = env.action_space.sample()  # Random action

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            logging.debug(f"Episode terminated with reward: {reward}")
            obs, _ = env.reset()
    
    print(f"\n{'*'*20}*")
    logging.info("✅ Simulation complete!")
    print(f"{'*'*80}*")
    print(f"{'*'*80}*\n")
    
    # Close environment
    env.close()