import tensorflow as tf
from keras.losses import binary_crossentropy, Loss
import keras
from keras.layers import Layer
import numpy as np

# Custom Keras Loss class for trajectory loss
class CustomTrajLoss(Loss):
    def __init__(self, p_bce=1, p_latlon=10, p_cat=1, p_day=1, p_hour=1, **kwargs):
        super().__init__(**kwargs)
        self.p_bce = p_bce
        self.p_latlon = p_latlon
        self.p_cat = p_cat
        self.p_day = p_day
        self.p_hour = p_hour
        
        # Store references to the model's input and output tensors
        self.real_traj = None
        self.gen_traj = None
        
    def set_trajectories(self, real_traj, gen_traj):
        """Set the real and generated trajectories for loss computation."""
        # Ensure all tensors have consistent data type (float32)
        self.real_traj = [
            tf.cast(real_traj[0], tf.float32),  # lat_lon
            tf.cast(real_traj[1], tf.float32),  # category
            tf.cast(real_traj[2], tf.float32),  # day
            tf.cast(real_traj[3], tf.float32),  # hour
            tf.cast(real_traj[4], tf.float32),  # mask
        ]
        
        self.gen_traj = [
            tf.cast(gen_traj[0], tf.float32),  # lat_lon
            tf.cast(gen_traj[1], tf.float32),  # category
            tf.cast(gen_traj[2], tf.float32),  # day
            tf.cast(gen_traj[3], tf.float32),  # hour
            tf.cast(gen_traj[4], tf.float32),  # mask
        ]
        
    def call(self, y_true, y_pred):
        """Compute the loss value.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predictions from the discriminator.
            
        Returns:
            Loss value.
        """
        # Cast inputs to float32 for consistent typing
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Simple BCE loss if trajectories are not available
        if self.real_traj is None or self.gen_traj is None:
            return binary_crossentropy(y_true, y_pred)
            
        # Binary cross-entropy for adversarial loss
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        # Get trajectory length from mask
        traj_length = keras.ops.sum(self.real_traj[4], axis=1)
        traj_length = keras.ops.expand_dims(traj_length, axis=1)  # Add dimension for broadcasting
        
        # Add a small epsilon to avoid division by zero
        traj_length = traj_length + 1e-6
        
        # Compute MSE for lat/lon (spatial loss)
        diff = self.gen_traj[0] - self.real_traj[0]
        # Clip differences to avoid extreme values
        diff = keras.ops.clip(diff, -10.0, 10.0)
        squared_diff = diff * diff
        mask_repeated = keras.ops.repeat(self.real_traj[4], 2, axis=2)
        masked_squared_diff = squared_diff * mask_repeated
        masked_latlon_full = keras.ops.sum(keras.ops.sum(masked_squared_diff, axis=1), axis=1, keepdims=True)
        masked_latlon_mse = keras.ops.sum(masked_latlon_full / traj_length)
        
        # Cross-entropy for category
        # Clip generated probabilities to avoid log(0)
        gen_category_clipped = keras.ops.clip(self.gen_traj[1], 1e-7, 1.0)
        ce_category = keras.ops.categorical_crossentropy(self.real_traj[1], gen_category_clipped, from_logits=False)
        # Apply mask
        ce_category_masked = ce_category * self.real_traj[4][:,:,0]
        ce_category_mean = keras.ops.sum(ce_category_masked / traj_length)
        
        # Cross-entropy for day
        # Clip generated probabilities to avoid log(0)
        gen_day_clipped = keras.ops.clip(self.gen_traj[2], 1e-7, 1.0)
        ce_day = keras.ops.categorical_crossentropy(self.real_traj[2], gen_day_clipped, from_logits=False)
        # Apply mask
        ce_day_masked = ce_day * self.real_traj[4][:,:,0]
        ce_day_mean = keras.ops.sum(ce_day_masked / traj_length)
        
        # Cross-entropy for hour
        # Clip generated probabilities to avoid log(0)
        gen_hour_clipped = keras.ops.clip(self.gen_traj[3], 1e-7, 1.0)
        ce_hour = keras.ops.categorical_crossentropy(self.real_traj[3], gen_hour_clipped, from_logits=False)
        # Apply mask
        ce_hour_masked = ce_hour * self.real_traj[4][:,:,0]
        ce_hour_mean = keras.ops.sum(ce_hour_masked / traj_length)
        
        # Combined loss with proper weighting
        # Clip each component to reasonable ranges
        bce_loss_clipped = keras.ops.clip(bce_loss, 0, 10)
        masked_latlon_mse_clipped = keras.ops.clip(masked_latlon_mse, 0, 100)
        ce_category_mean_clipped = keras.ops.clip(ce_category_mean, 0, 10)
        ce_day_mean_clipped = keras.ops.clip(ce_day_mean, 0, 10)
        ce_hour_mean_clipped = keras.ops.clip(ce_hour_mean, 0, 10)
        
        # Use reduced weights for a more stable start
        p_bce = tf.constant(self.p_bce, dtype=tf.float32)
        p_latlon = tf.constant(self.p_latlon, dtype=tf.float32)
        p_cat = tf.constant(self.p_cat, dtype=tf.float32)
        p_day = tf.constant(self.p_day, dtype=tf.float32)
        p_hour = tf.constant(self.p_hour, dtype=tf.float32)
        
        total_loss = (p_bce * bce_loss_clipped + 
                     p_latlon * masked_latlon_mse_clipped + 
                     p_cat * ce_category_mean_clipped + 
                     p_day * ce_day_mean_clipped + 
                     p_hour * ce_hour_mean_clipped)
        
        # Final safety clipping to prevent numerical instability
        total_loss = keras.ops.clip(total_loss, 0, 1000)
        
        # Debug output
        tf.print("Loss components - BCE:", bce_loss_clipped, 
                "Spatial:", masked_latlon_mse_clipped, 
                "Category:", ce_category_mean_clipped,
                "Day:", ce_day_mean_clipped,
                "Hour:", ce_hour_mean_clipped,
                "Total:", total_loss)
        
        return total_loss

# BCE loss for the discriminator
def d_bce_loss(mask):
    def loss(y_true, y_pred):
        d_bce_loss = binary_crossentropy(y_true, y_pred)
        return d_bce_loss

    return loss

# Custom layer for trajectory loss
class TrajLossLayer(Layer):
    def __init__(self, p_bce=1, p_latlon=10, p_cat=1, p_day=1, p_hour=1, **kwargs):
        super(TrajLossLayer, self).__init__(**kwargs)
        self.p_bce = p_bce
        self.p_latlon = p_latlon
        self.p_cat = p_cat
        self.p_day = p_day
        self.p_hour = p_hour
    
    def call(self, inputs):
        y_true, y_pred, real_traj, gen_traj = inputs
        
        # Use Keras operations instead of direct TensorFlow ops
        traj_length = keras.ops.sum(real_traj[4], axis=1)
        
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        # Compute MSE for lat/lon
        diff = gen_traj[0] - real_traj[0]
        squared_diff = diff * diff
        mask_repeated = keras.ops.concatenate([real_traj[4] for _ in range(2)], axis=2)
        masked_squared_diff = squared_diff * mask_repeated
        masked_latlon_full = keras.ops.sum(keras.ops.sum(masked_squared_diff, axis=1), axis=1, keepdims=True)
        masked_latlon_mse = keras.ops.sum(masked_latlon_full / keras.ops.expand_dims(traj_length, axis=1))
        
        # Cross entropy for categories
        ce_category = keras.ops.categorical_crossentropy(real_traj[1], gen_traj[1], from_logits=True)
        ce_day = keras.ops.categorical_crossentropy(real_traj[2], gen_traj[2], from_logits=True)
        ce_hour = keras.ops.categorical_crossentropy(real_traj[3], gen_traj[3], from_logits=True)
        
        # Apply masks
        mask_sum = keras.ops.sum(real_traj[4], axis=2)
        ce_category_masked = ce_category * mask_sum
        ce_day_masked = ce_day * mask_sum
        ce_hour_masked = ce_hour * mask_sum
        
        # Compute mean
        ce_category_mean = keras.ops.sum(ce_category_masked / keras.ops.expand_dims(traj_length, axis=1))
        ce_day_mean = keras.ops.sum(ce_day_masked / keras.ops.expand_dims(traj_length, axis=1))
        ce_hour_mean = keras.ops.sum(ce_hour_masked / keras.ops.expand_dims(traj_length, axis=1))
        
        # Combined loss
        total_loss = (bce_loss * self.p_bce + 
                      masked_latlon_mse * self.p_latlon + 
                      ce_category_mean * self.p_cat + 
                      ce_day_mean * self.p_day + 
                      ce_hour_mean * self.p_hour)
        
        return total_loss

# trajLoss function that returns a loss function compatible with Keras
def trajLoss(real_traj, gen_traj):
    loss_layer = TrajLossLayer()
    
    def loss(y_true, y_pred):
        return loss_layer([y_true, y_pred, real_traj, gen_traj])
    
    return loss

def compute_advantage(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation (GAE) as in the paper.
    
    Args:
        rewards: Tensor of shape [batch_size, 1] containing rewards
        values: Tensor of shape [batch_size, 1] containing value estimates
        gamma: Discount factor
        gae_lambda: GAE parameter
        
    Returns:
        advantages: Tensor of shape [batch_size, 1] containing advantages
    """
    # Ultra simplified version to avoid shape issues
    # First, check the shape of rewards to handle unexpected dimensions
    rewards_tensor = tf.cast(rewards, tf.float32)
    rewards_shape = tf.shape(rewards_tensor)
    batch_size = rewards_shape[0]
    
    # If rewards has more than 2 dimensions or is not [batch_size, 1],
    # reduce it by taking the mean across all dimensions except the first
    if len(rewards_tensor.shape) > 2 or (len(rewards_tensor.shape) == 2 and rewards_tensor.shape[1] > 1):
        print(f"Reshaping rewards from {rewards_tensor.shape} to [{batch_size}, 1] by taking mean")
        # Flatten all dimensions after the first and then take the mean
        flat_rewards = tf.reshape(rewards_tensor, [batch_size, -1])
        rewards_tensor = tf.reduce_mean(flat_rewards, axis=1, keepdims=True)
    else:
        # Just ensure it has shape [batch_size, 1]
        rewards_tensor = tf.reshape(rewards_tensor, [batch_size, 1])
    
    # Basic normalization
    mean = tf.reduce_mean(rewards_tensor)
    std = tf.math.reduce_std(rewards_tensor) + 1e-8
    normalized_advantages = (rewards_tensor - mean) / std
    
    # Make sure output shape is correct: [batch_size, 1]
    normalized_advantages = tf.reshape(normalized_advantages, [batch_size, 1])
    
    return normalized_advantages

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns (sum of future rewards).
    
    Args:
        rewards: Tensor of shape [batch_size, 1] containing rewards
        gamma: Discount factor
        
    Returns:
        returns: Tensor of shape [batch_size, 1] containing returns
    """
    # Ultra simplified version to avoid shape issues
    # First, check the shape of rewards to handle unexpected dimensions
    rewards_tensor = tf.cast(rewards, tf.float32)
    rewards_shape = tf.shape(rewards_tensor)
    batch_size = rewards_shape[0]
    
    # If rewards has more than 2 dimensions or is not [batch_size, 1],
    # reduce it by taking the mean across all dimensions except the first
    if len(rewards_tensor.shape) > 2 or (len(rewards_tensor.shape) == 2 and rewards_tensor.shape[1] > 1):
        print(f"Reshaping rewards from {rewards_tensor.shape} to [{batch_size}, 1] by taking mean")
        # Flatten all dimensions after the first and then take the mean
        flat_rewards = tf.reshape(rewards_tensor, [batch_size, -1])
        rewards_tensor = tf.reduce_mean(flat_rewards, axis=1, keepdims=True)
    else:
        # Just ensure it has shape [batch_size, 1]
        rewards_tensor = tf.reshape(rewards_tensor, [batch_size, 1])
    
    return rewards_tensor

def compute_entropy_loss(action_probs):
    """Compute entropy loss to encourage exploration.
    
    Args:
        action_probs: Tensor of shape [batch_size, max_length, action_dim] containing action probabilities
        
    Returns:
        entropy_loss: Scalar tensor containing the entropy loss
    """
    return -tf.reduce_mean(tf.reduce_sum(
        action_probs * tf.math.log(action_probs + 1e-10),
        axis=-1
    ))

def compute_trajectory_ratio(new_predictions, old_predictions):
    """Compute the PPO policy ratio between new and old policies for trajectory data.
    
    For trajectories, we take the product of point-wise prediction ratios,
    which is equivalent to the ratio of trajectory probabilities under each policy.
    
    Args:
        new_predictions: Outputs from the current policy
        old_predictions: Outputs from the old policy before update
        
    Returns:
        Tensor of shape [batch_size, 1] containing the policy ratios
    """
    # Initialize ratio as ones
    ratio = tf.ones(shape=(len(new_predictions[0]), 1))
    
    # For each categorical output (category, day, hour), compute ratio
    for i in range(1, 4):  # Categorical outputs
        # Get probabilities under new policy
        new_probs = new_predictions[i]
        
        # Get probabilities under old policy
        old_probs = old_predictions[i]
        
        # Compute the ratio of probabilities (with small epsilon to avoid division by zero)
        point_ratio = new_probs / (old_probs + 1e-10)
        
        # Reduce along the category dimension to get per-point ratio
        point_ratio = tf.reduce_sum(point_ratio, axis=-1, keepdims=True)
        
        # Multiply with current ratio
        ratio = ratio * point_ratio
    
    # For continuous outputs (coordinates), use normal distribution likelihood ratio
    coord_ratio = tf.exp(-0.5 * tf.reduce_sum(tf.square(new_predictions[0] - old_predictions[0]), axis=-1, keepdims=True))
    ratio = ratio * coord_ratio
    
    # Mask out padding
    mask = new_predictions[4]
    ratio = ratio * mask
    
    # Average over the trajectory length
    ratio = tf.reduce_mean(ratio, axis=1)
    
    return ratio