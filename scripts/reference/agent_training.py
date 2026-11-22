"""
Train a robotic pushing agent using Stable Baselines 3 SAC algorithm.
This script handles training, evaluation, logging, and visualization.
"""

import os
import time
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

# Stable Baselines 3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, 
    EvalCallback, CallbackList
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

# Import agent environment
if __package__:
    from .agent_environment import RoboticPushingEnv
else:
    from agent_environment import RoboticPushingEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except ImportError:
    tensorboard_available = False
    print("TensorBoard not available. Install with: pip install tensorboard")

class CurriculumCallback(BaseCallback):
    def __init__(self, env, curriculum_ranges, eval_freq=50, success_threshold=0.8, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.curriculum_ranges = curriculum_ranges
        self.current_level = 0
        self.eval_freq = eval_freq
        self.success_threshold = success_threshold

        self.episodes_count = 0
        self.episode_successes = []

        # For detailed monitoring
        self.level_durations = [0] * len(curriculum_ranges)
        self.level_transitions = []

        # Level descriptions for better logging
        self.level_names = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]

        # Ensure we have names for all levels
        if len(self.level_names) < len(curriculum_ranges):
            for i in range(len(self.level_names), len(curriculum_ranges)):
                self.level_names.append(f"Level {i+1}")

    def _on_training_start(self) -> None:
        """
        Initialize TensorBoard logging when training starts
        """
        # Log initial curriculum level
        self.logger.record("curriculum/level", self.current_level)
        self.logger.record("curriculum/level_name",
                           self.level_names[self.current_level])

        # Log initial ranges
        x_range, y_range = self.curriculum_ranges[self.current_level]
        self.logger.record("curriculum/x_min", x_range[0])
        self.logger.record("curriculum/x_max", x_range[1])
        self.logger.record("curriculum/y_min", y_range[0])
        self.logger.record("curriculum/y_max", y_range[1])

        print(
            f"[Curriculum] Starting with {self.level_names[self.current_level]}: X={x_range}, Y={y_range}")

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals.get("dones", [False])[-1]
        info = self.locals.get("infos", [{}])[-1]

        if done:
            self.episodes_count += 1
            is_success = info.get("is_success", False)
            self.episode_successes.append(is_success)

            # Increment counter for current level duration
            self.level_durations[self.current_level] += 1

            # Every eval_freq episodes, check if we can bump difficulty
            if self.episodes_count % self.eval_freq == 0:
                # Calculate recent success rate
                recent_success = sum(
                    self.episode_successes[-self.eval_freq:]) / self.eval_freq

                # Log the success rate to TensorBoard
                self.logger.record("curriculum/success_rate", recent_success)

                if self.verbose > 0:
                    print(
                        f"[Curriculum] {self.eval_freq} episodes complete. Success rate = {recent_success:.2f}")

                # If success is high, move up the curriculum list
                previous_level = self.current_level
                if recent_success >= self.success_threshold:
                    self.current_level = min(
                        self.current_level + 1,
                        len(self.curriculum_ranges) - 1
                    )

                    # Record the level transition
                    if previous_level != self.current_level:
                        timestamp = self.num_timesteps
                        self.level_transitions.append({
                            'timestep': timestamp,
                            'episode': self.episodes_count,
                            'from_level': previous_level,
                            'to_level': self.current_level,
                            'success_rate': recent_success
                        })

                        # Log level change to TensorBoard
                        self.logger.record(
                            "curriculum/level", self.current_level)
                        self.logger.record(
                            "curriculum/level_name", self.level_names[self.current_level])

                (new_x_range,
                 new_y_range) = self.curriculum_ranges[self.current_level]

                # Log current ranges to TensorBoard regardless of change
                self.logger.record("curriculum/x_min", new_x_range[0])
                self.logger.record("curriculum/x_max", new_x_range[1])
                self.logger.record("curriculum/y_min", new_y_range[0])
                self.logger.record("curriculum/y_max", new_y_range[1])

                # Log level durations periodically
                for i, duration in enumerate(self.level_durations):
                    self.logger.record(
                        f"curriculum/level_{i}_episodes", duration)

                # Only print message if level changed
                if previous_level != self.current_level:
                    if self.verbose > 0:
                        print(
                            f"[Curriculum] 🔼 Upgrading from {self.level_names[previous_level]} to {self.level_names[self.current_level]}")
                        print(
                            f"[Curriculum] New ranges: X={new_x_range}, Y={new_y_range}")

                # If you have a VecEnv, apply to each sub-env
                if hasattr(self.env, 'envs'):
                    for sub_env in self.env.envs:
                        sub_env.object_spawn_range_x = new_x_range
                        sub_env.object_spawn_range_y = new_y_range
                else:
                    # Single environment
                    self.env.object_spawn_range_x = new_x_range
                    self.env.object_spawn_range_y = new_y_range

        return True

    def _on_training_end(self) -> None:
        """
        Final logging at the end of training
        """
        # Print summary of curriculum progression
        print("\n" + "="*50)
        print("CURRICULUM LEARNING SUMMARY")
        print("="*50)
        print(f"Total episodes: {self.episodes_count}")
        print(
            f"Final curriculum level: {self.level_names[self.current_level]}")
        print("\nLevel durations (episodes):")
        for i, duration in enumerate(self.level_durations):
            if duration > 0:
                print(
                    f"  {self.level_names[i]}: {duration} episodes ({duration/self.episodes_count*100:.1f}%)")

        print("\nLevel transitions:")
        for t in self.level_transitions:
            print(
                f"  Episode {t['episode']}: {self.level_names[t['from_level']]} → {self.level_names[t['to_level']]} (Success rate: {t['success_rate']:.2f})")
        print("="*50)



class RobotPushTrainer:
    """
    Trainer class for robotic pushing using Stable Baselines 3.
    
    This class encapsulates the environment setup, model configuration,
    training, evaluation, and visualization functionality.
    """
    
    def __init__(
        self,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_steps: int = 1,
        train_freq: int = 1,
        learning_starts: int = 10000,
        use_sde: bool = True,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        ent_coef: Union[str, float] = "auto",
        checkpoint_freq: int = 10000,
        eval_freq: int = 10000,
        random_seed: Optional[int] = None,
        n_envs: int = 1,
        parallel_envs: bool = False,
        max_steps: int = 1000,
        load_model: Optional[str] = None,
        log_dir: str = "logs",
        model_dir: str = "models",
        render_mode: Optional[str] = None,
        difficulty_level: int = 0,
        observation_type: str = "multimodal",
        eval_only: bool = False,
        *,
        trust_checkpoint: bool = False,
    ):
        """
        Initialize the trainer with configuration parameters.
        
        Args:
            total_timesteps: Total timesteps for training
            learning_rate: Learning rate for the optimizer
            buffer_size: Size of the replay buffer
            batch_size: Batch size for each gradient update
            gamma: Discount factor
            tau: Target network update rate
            gradient_steps: How many gradient steps to take per update
            train_freq: Update the model every train_freq steps
            learning_starts: How many steps to collect before learning starts
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: Sample frequency for SDE
            tensorboard_log: TensorBoard log directory
            ent_coef: Entropy coefficient or 'auto'
            checkpoint_freq: Frequency of checkpoints
            eval_freq: Frequency of evaluations
            random_seed: Random seed for reproducibility
            n_envs: Number of parallel environments
            parallel_envs: Whether to use SubprocVecEnv (parallel) or DummyVecEnv
            max_steps: Maximum number of steps per episode
            load_model: Path to a pre-trained model to load
            log_dir: Directory for logs
            model_dir: Directory for model saves
            render_mode: Rendering mode for the environment
            difficulty_level: Difficulty level for the object spawn range
            observation_type: Type of observation space (multimodal, multimodal_no_goal, vision_only, proprio_only)
        """
        # Store configuration parameters
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.tensorboard_log = tensorboard_log
        self.ent_coef = ent_coef
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.random_seed = random_seed
        self.n_envs = n_envs
        self.parallel_envs = parallel_envs
        self.max_steps = max_steps
        self.load_model = load_model
        self.trust_checkpoint = trust_checkpoint
        if self.load_model and self.trust_checkpoint is not True:
            raise ValueError("Loading a checkpoint requires trust_checkpoint=True.")
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.render_mode = render_mode
        self.difficulty_level = difficulty_level
        self.observation_type = observation_type
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            # TODO: add torch also?
            
        # Initialize environment and model
        self.env = None
        self.eval_env = None
        self.model = None
        
        # Setup additional variables
        self.run_id = f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure logger
        self.logger_path = os.path.join(self.log_dir, self.run_id)
        os.makedirs(self.logger_path, exist_ok=True)

        # Curriculum learning object positions ranges
        self.curriculum_ranges = [
            ((0.58, 0.60), (-0.05, 0.05)),  # Very easy
            ((0.55, 0.60), (-0.07, 0.07)),    # Easy
            ((0.52, 0.60), (-0.10, 0.10)),    # Medium
            ((0.50, 0.60), (-0.12, 0.12)),   # Hard
            ((0.48, 0.60), (-0.15, 0.15)),    # Very hard
        ]

        self.curriculum_ranges_eval = [
            ((0.58, 0.58), (-0.05, -0.05)),  # Very easy
            ((0.55, 0.55), (0.07, 0.07)),    # Easy
            ((0.52, 0.52), (-0.10, -0.10)),    # Medium
            ((0.50, 0.50), (0.12, 0.12)),   # Hard
            ((0.48, 0.48), (-0.15, -0.15)),    # Very hard
        ]

        # Level descriptions for better logging
        self.level_names = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]

        # Check valid difficulty level
        total_levels = len(self.curriculum_ranges)
        if self.difficulty_level < 0 or self.difficulty_level >= total_levels:
            raise ValueError(f"Invalid difficulty level. Should be between 0 and {total_levels-1}")

        # Map difficulty levels (int: 0 to 4) to curriculum ranges (very easy to very hard)
        if eval_only:
            self.object_spawn_range_x, self.object_spawn_range_y = self.curriculum_ranges_eval[self.difficulty_level]
        else:
            self.object_spawn_range_x, self.object_spawn_range_y = self.curriculum_ranges[self.difficulty_level]

        print(f"[INFO] Difficulty level {self.difficulty_level}: {self.level_names[self.difficulty_level]}")
        print(f"[INFO] Object spawn range x: {self.object_spawn_range_x}")
        print(f"[INFO] Object spawn range y: {self.object_spawn_range_y}")

        if tensorboard_log:
            self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log)
        
    def setup_environments(self):
        """Create and configure training and evaluation environments."""
        print("Setting up environments...")

        # Store parameters locally to avoid capturing self in closures
        render_mode = self.render_mode
        max_steps = self.max_steps
        logger_path = self.logger_path
        object_spawn_range_x = self.object_spawn_range_x
        object_spawn_range_y = self.object_spawn_range_y
        observation_type = self.observation_type
        
        # Configure environment creation function
        def make_env(rank):
            def _init():
                env = RoboticPushingEnv(
                    render_mode=render_mode if rank == 0 else None,
                    max_steps=max_steps,
                    object_spawn_range_x=object_spawn_range_x,
                    object_spawn_range_y=object_spawn_range_y,
                    observation_type=observation_type
                )
                # Add Monitor wrapper for logging episode statistics
                env = Monitor(env, os.path.join(logger_path, f"env_{rank}"))
                return env
            return _init
        
        # Create vectorized environment for training
        if self.n_envs > 1:
            if self.parallel_envs:
                self.env = SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
            else:
                self.env = DummyVecEnv([make_env(i) for i in range(self.n_envs)])
        else:
            self.env = DummyVecEnv([make_env(0)])
        
        # Create separate environment for evaluation
        self.eval_env = DummyVecEnv([make_env(0)])
        
        print(f"Created training environment with {self.n_envs} {'parallel' if self.parallel_envs else 'sequential'} environments")
        
    def setup_model(self):
        """Set up the SAC model with specified hyperparameters."""
        print("Setting up SAC model...")
        
        # Load pre-trained model if specified
        if self.load_model:
            print(f"Loading pre-trained model from {self.load_model}")
            self.model = self._load_checkpoint(
                self.load_model,
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                gradient_steps=self.gradient_steps,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
                use_sde=self.use_sde,
                sde_sample_freq=self.sde_sample_freq,
                ent_coef=self.ent_coef,
                verbose=1,
                seed=self.random_seed,
            )
            # Reset replay buffer if you want to start fresh for fine-tuning
            # self.model.replay_buffer = None
        else:
            # Create a new model
            self.model = SAC(
                "MultiInputPolicy",  # Required for dict observation spaces
                self.env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                gradient_steps=self.gradient_steps,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
                use_sde=self.use_sde,
                sde_sample_freq=self.sde_sample_freq,
                tensorboard_log=self.tensorboard_log,
                ent_coef=self.ent_coef,
                verbose=1,
                seed=self.random_seed,
            )
            
        print("Model setup complete")
        
    def setup_callbacks(self):
        """Set up training callbacks for checkpointing, evaluation, etc."""
        print("Setting up callbacks...")
        
        callbacks = []
        
        # Checkpoint callback saves the model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=self.model_dir,
            name_prefix=f"sac_pusher_{self.run_id}",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback periodically evaluates the model
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_model"),
            log_path=self.logger_path,
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)

        # Curriculum callback for adjusting difficulty level
        # Step 1: define or pass your curriculum ranges
        # self.curriculum_ranges

        # Step 2: make the callback
        curriculum_callback = CurriculumCallback(
            env=self.env,                 # or whatever variable references the training env
            curriculum_ranges=self.curriculum_ranges,
            eval_freq=50,                 # check every 50 episodes
            success_threshold=0.8,
            verbose=1
        )

        callbacks.append(curriculum_callback)
        
        return CallbackList(callbacks)
    
    def train(self):
        """Train the SAC agent."""
        print(f"Starting training for {self.total_timesteps} timesteps...")
        
        # Setup environments if not done already
        if self.env is None:
            self.setup_environments()
            
        # Setup model if not done already
        if self.model is None:
            self.setup_model()
            
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        try:
            # Start training
            start_time = time.time()
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                log_interval=10,
                tb_log_name=self.run_id,
                reset_num_timesteps=True,
            )
            training_time = time.time() - start_time
            
            # Save the final model
            final_model_path = os.path.join(self.model_dir, f"final_model_{self.run_id}")
            self.model.save(final_model_path)
            print(f"Training completed in {training_time:.2f} seconds.")
            print(f"Final model saved to {final_model_path}")
            
            return final_model_path
            
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            # Save the model on interrupt
            interrupted_model_path = os.path.join(self.model_dir, f"interrupted_model_{self.run_id}")
            self.model.save(interrupted_model_path)
            print(f"Interrupted model saved to {interrupted_model_path}")
            return interrupted_model_path
    
    def _load_checkpoint(self, path, **kwargs):
        if self.trust_checkpoint is not True:
            raise ValueError("Loading a checkpoint requires trust_checkpoint=True.")
        return SAC.load(path, **kwargs)

    def evaluate(self, model_path=None, n_eval_episodes=10, deterministic=True):
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to the model to evaluate (if None, uses the current model)
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            mean_reward, std_reward: Mean and standard deviation of episode rewards
        """
        print(f"Evaluating model over {n_eval_episodes} episodes...")
        
        # Setup environments if not done already
        if self.eval_env is None:
            self.setup_environments()
        
        # Load model if path is provided
        if model_path:
            model = self._load_checkpoint(model_path, env=self.eval_env)
        else:
            # Use existing model
            model = self.model
            
        if model is None:
            raise ValueError("No model available for evaluation. Either provide a model_path or train a model first.")
        
        # Run evaluation
        mean_reward, std_reward = evaluate_policy(
            model, 
            self.eval_env, 
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=True if self.render_mode else False,
            return_episode_rewards=False,
        )
        
        print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward
    
    def run_episode(self, model_path=None, deterministic=True, render=True, record=False, video_path=None, fps=10):
        """
        Run a single episode with the trained model (useful for visualization).
        
        Args:
            model_path: Path to the model to run (if None, uses the current model)
            deterministic: Whether to use deterministic actions
            render: Whether to render the episode
            record: Whether to record a video of the episode
            video_path: Path to save the video (default: "episode_recording.mp4")
            fps: Frames per second for the output video
        """

        import cv2
        import pybullet as p
        import numpy as np

        # Create a dedicated environment for running episodes
        env = RoboticPushingEnv(
            render_mode="human" if render else None,
            max_steps=self.max_steps,
            object_spawn_range_x=self.object_spawn_range_x,
            object_spawn_range_y=self.object_spawn_range_y,
            observation_type=self.observation_type
        )

        # Video recording setup
        video_writer = None
        if record:
            if video_path is None:
                video_path = f"outputs/videos/push_episode_{int(time.time())}.mp4"
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
            print(f"Recording video to {video_path}")
        
        # Load model if path is provided
        if model_path:
            model = self._load_checkpoint(model_path)
        else:
            # Use existing model
            model = self.model
            
        if model is None:
            raise ValueError("No model available. Either provide a model_path or train a model first.")
        
        # Reapply the caller's seed after loading checkpoint metadata.
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print("Running episode...")
        
        # Create a list to store distance metrics over time
        distances = []
        
        while True:
            # Capture the initial state and every state through termination.
            if record:
                # Get the camera image
                width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                    width=1280,  # HD resolution
                    height=720,
                    viewMatrix=env.view_matrix,
                    projectionMatrix=env.projection_matrix,
                    physicsClientId=env.physics_client_id
                )
                
                # Convert RGB buffer to a proper image format
                rgb_array = np.reshape(rgb_img, (height, width, 4))[:,:,:3]
                
                # Initialize video writer on first frame
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, fps, (width, height)
                    )
                    if not video_writer.isOpened():
                        env.close()
                        raise OSError(f"Could not open video for writing: {video_path}")
                
                # Convert RGB to BGR for OpenCV
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                
                # Add frame to video
                video_writer.write(bgr_array)
            
            if done or truncated:
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Calculate distance between object center and target
            if hasattr(env, 'last_detected_bbox') and hasattr(env, 'target_position_2d'):
                bbox = env.last_detected_bbox
                obj_center_x = bbox[0] + bbox[2]/2
                obj_center_y = bbox[1] + bbox[3]/2
                
                target_x = env.target_position_2d[0]
                target_y = env.target_position_2d[1]
                
                dist = np.sqrt((obj_center_x - target_x)**2 + (obj_center_y - target_y)**2)
                
                # Store distance and print current distance
                distances.append(dist)
                print(f"Step {step_count}: Distance to target = {dist:.4f}, Reward = {reward:.4f}")
            
            if render:
                time.sleep(0.01)  # Slow down for visibility

        # Clean up video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_path}")

        # Print episode results
        print(f"\n*******************\n")
        print(f"Episode finished after {step_count} steps with total reward: {total_reward:.2f}")
        print(f"Success: {info.get('is_success', False)}")
        print(f"Termination reason: {info.get('termination_reason', 'unknown')}")
        
        # Print distance statistics
        if distances:
            print(f"Distance Statistics:")
            print(f"  Initial distance: {distances[0]:.4f}")
            print(f"  Final distance: {distances[-1]:.4f}")
            print(f"  Minimum distance: {min(distances):.4f}")
            print(f"  Average distance: {sum(distances)/len(distances):.4f}")
            
            # Check if the distance threshold is available from the environment
            if hasattr(env, 'distance_threshold'):
                print(f"  Distance threshold for success: {env.distance_threshold:.4f}")
        
        print(f"\n*******************\n")

        if render:
            print("Press 'q' to quit or Ctrl+C to exit...")
            try:
                while input().strip().lower() != "q":
                    pass
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
        
        env.close()
        return total_reward, info
        

def parse_arguments():
    """Parse command line arguments for the trainer."""
    parser = argparse.ArgumentParser(description="Train a robotic pushing agent with SAC")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps per update")
    parser.add_argument("--train-freq", type=int, default=1, help="Update frequency")
    parser.add_argument("--learning-starts", type=int, default=10000, help="Steps before learning starts")
    parser.add_argument("--ent-coef", type=str, default="auto", help="Entropy coefficient ('auto' or float)")
    
    # Environment parameters
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--parallel", action="store_true", help="Use parallel environments (SubprocVecEnv)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Open the simulator window during training or recording")
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for model saves")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard logging")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (in timesteps)")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--load-model", type=str, default=None, help="Path to model to load")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow loading a checkpoint from a trusted source (can execute code)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on a trained model")
    parser.add_argument("--run-episode", action="store_true", help="Run a single episode with a trained model")
    parser.add_argument("--randomize-spawn", action="store_true",
                        help="Sample a spawn from the difficulty range during single-episode playback")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--difficulty-level", type=int, default=0, help="Difficulty level for the object spawn range (between 0 to 4)")
    parser.add_argument("--observation-type", type=str, default="multimodal", help="Observation type multimodal, multimodal_no_goal, vision_only, proprio_only")

    # Recording options
    parser.add_argument("--record", action="store_true", help="Record a video of the episode")
    parser.add_argument("--video-path", type=str, default=None, help="Path to save the video file")
    
    args = parser.parse_args()
    if args.load_model and not args.trust_checkpoint:
        parser.error("Loading a checkpoint requires --trust-checkpoint (trusted sources only).")
    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    ent_coef = "auto_0.1"
    
    # Configure TensorBoard logging
    tensorboard_log = None
    if args.tensorboard and tensorboard_available:
        tensorboard_log = os.path.join(args.log_dir, "tensorboard")
        os.makedirs(tensorboard_log, exist_ok=True)
    
    # Initialize trainer
    trainer = RobotPushTrainer(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        gradient_steps=args.gradient_steps,
        train_freq=args.train_freq,
        learning_starts=args.learning_starts,
        tensorboard_log=tensorboard_log,
        ent_coef=ent_coef,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        random_seed=args.seed,
        n_envs=args.n_envs,
        parallel_envs=args.parallel,
        max_steps=args.max_steps,
        load_model=args.load_model,
        trust_checkpoint=args.trust_checkpoint,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        render_mode="human" if args.render else None,
        difficulty_level=args.difficulty_level,
        observation_type=args.observation_type,
        eval_only=args.run_episode and not args.randomize_spawn
    )
    
    # Mode selection based on command line arguments
    if args.eval_only and args.load_model:
        # Evaluation mode
        trainer.evaluate(
            model_path=args.load_model,
            n_eval_episodes=args.n_eval_episodes
        )
    elif args.run_episode and args.load_model:
        # Run a single episode
        trainer.run_episode(
            model_path=args.load_model,
            deterministic=True,
            render=args.render or not args.record,
            record=args.record,
            video_path=args.video_path
        )
    else:
        # Training mode
        trainer.train()
        
        # Optionally evaluate the trained model
        trainer.evaluate(
            n_eval_episodes=args.n_eval_episodes
        )
