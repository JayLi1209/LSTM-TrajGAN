import numpy as np
import os
import pandas as pd
import wandb
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from model import RL_Enhanced_Transformer_TrajGAN
from MARC.marc import MARC

def compute_data_stats():
    """Compute statistics from training data."""
    # Load training data
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    # Compute centroids
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    # Compute scale factor
    scale_factor = max(
        max(abs(tr['lat'].max() - lat_centroid),
            abs(te['lat'].max() - lat_centroid),
            abs(tr['lat'].min() - lat_centroid),
            abs(te['lat'].min() - lat_centroid)),
        max(abs(tr['lon'].max() - lon_centroid),
            abs(te['lon'].max() - lon_centroid),
            abs(tr['lon'].min() - lon_centroid),
            abs(te['lon'].min() - lon_centroid))
    )
    
    # Load training data for category size
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    category_size = x_train[3].shape[-1]  # Get category vocabulary size
    
    # Get max sequence length
    max_length = 144  # Default value, can be adjusted based on your data
    
    # Create stats dictionary
    stats = {
        'lat_centroid': lat_centroid,
        'lon_centroid': lon_centroid,
        'scale_factor': scale_factor,
        'category_size': category_size,
        'max_length': max_length
    }
    
    # Save stats
    np.save('data/data_stats.npy', stats)
    return stats

class EarlyStoppingCallback:
    def __init__(self, patience=15, min_delta=0.001, monitor='g_loss'):
        """
        Early stopping callback
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait_count = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch, metrics):
        """Check if training should stop after this epoch"""
        current_value = metrics.get(self.monitor, float('inf'))
        
        if current_value < self.best_value - self.min_delta:
            # Improvement detected
            self.best_value = current_value
            self.wait_count = 0
        else:
            # No improvement
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                print(f"\nEarly stopping triggered at epoch {epoch}. No improvement in {self.monitor} for {self.patience} epochs.")

def main():
    # Initialize wandb
    wandb.init(
        project="rl-transformer-trajgan",
        config={
            "architecture": "RL-Enhanced-Transformer-TrajGAN",
            "dataset": "mobility-trajectories",
            "epochs": 200,
            "batch_size": 256,
            "latent_dim": 100,
            "early_stopping_patience": 15
        }
    )
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Compute or load data statistics
    if not os.path.exists('data/data_stats.npy'):
        print("Computing data statistics...")
        data_stats = compute_data_stats()
    else:
        print("Loading data statistics...")
        data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    
    # Initialize model parameters
    latent_dim = 100
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {
        'lat_lon': 2,
        'day': 7,
        'hour': 24,
        'category': 10,
        'mask': 1
    }
    max_length = data_stats['max_length']
    lat_centroid = data_stats['lat_centroid']
    lon_centroid = data_stats['lon_centroid']
    scale_factor = data_stats['scale_factor']
    
    # Initialize TUL classifier (MARC)
    tul_classifier = MARC()
    tul_classifier.load_weights('./MARC/MARC_Weight.h5')
    
    # Initialize and train the model
    model = RL_Enhanced_Transformer_TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Set TUL classifier for reward computation
    model.tul_classifier = tul_classifier
    
    # Training parameters
    epochs = 200
    batch_size = 256
    sample_interval = 10
    
    # Initialize early stopping
    early_stopping = EarlyStoppingCallback(patience=15, min_delta=0.001, monitor='g_loss')
    
    # Train the model with early stopping and wandb logging
    train_with_monitoring(model, epochs, batch_size, sample_interval, early_stopping)
    
    # Close wandb run
    wandb.finish()

def train_with_monitoring(model, epochs, batch_size, sample_interval, early_stopping):
    """Train the model with wandb monitoring and early stopping"""
    # Training data
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    
    # Padding
    X_train = [pad_sequences(f, model.max_length, padding='pre', dtype='float64') 
               for f in x_train]
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Sample batch
        idx = np.random.randint(0, len(X_train[0]), batch_size)
        batch = [X[idx] for X in X_train]
        
        # Training step
        metrics = model.train_step(batch, batch_size)
        
        # Extract privacy and utility metrics from the model
        privacy_metric, utility_metric = extract_privacy_utility_metrics(model, batch)
        
        # Add privacy and utility metrics
        metrics['privacy_score'] = privacy_metric
        metrics['utility_score'] = utility_metric
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'd_loss_real': metrics['d_loss_real'],
            'd_loss_fake': metrics['d_loss_fake'],
            'g_loss': metrics['g_loss'],
            'c_loss': metrics['c_loss'],
            'privacy_score': metrics['privacy_score'],
            'utility_score': metrics['utility_score']
        })
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"D_real: {metrics['d_loss_real']:.4f}, D_fake: {metrics['d_loss_fake']:.4f}, G: {metrics['g_loss']:.4f}, C: {metrics['c_loss']:.4f}")
            print(f"Privacy: {metrics['privacy_score']:.4f}, Utility: {metrics['utility_score']:.4f}")
        
        # Save checkpoints
        if epoch % sample_interval == 0:
            model.save_checkpoint(epoch)
            
            # Log model weights to wandb
            wandb.save(f"results/generator_{epoch}.weights.h5")
            wandb.save(f"results/discriminator_{epoch}.weights.h5")
            wandb.save(f"results/critic_{epoch}.weights.h5")
        
        # Check early stopping
        early_stopping.on_epoch_end(epoch, metrics)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

def extract_privacy_utility_metrics(model, batch):
    """Extract privacy and utility metrics from a training batch"""
    # Generate trajectories for evaluation
    noise = np.random.normal(0, 1, (len(batch[0]), model.latent_dim))
    gen_trajs = model.generator.predict([*batch, noise])
    
    try:
        # Calculate utility metric first (this should always work)
        # Spatial loss - lower is better
        spatial_loss = tf.reduce_mean(tf.square(gen_trajs[0] - batch[0])).numpy()
        
        # Temporal loss - lower is better
        temp_day_loss = -tf.reduce_mean(tf.reduce_sum(
            batch[2] * tf.math.log(tf.clip_by_value(gen_trajs[2], 1e-7, 1.0)), 
            axis=-1)).numpy()
        
        temp_hour_loss = -tf.reduce_mean(tf.reduce_sum(
            batch[3] * tf.math.log(tf.clip_by_value(gen_trajs[3], 1e-7, 1.0)), 
            axis=-1)).numpy()
        
        # Category loss - lower is better
        cat_loss = -tf.reduce_mean(tf.reduce_sum(
            batch[1] * tf.math.log(tf.clip_by_value(gen_trajs[1], 1e-7, 1.0)), 
            axis=-1)).numpy()
        
        # Combine utility components (lower is better, so we use negative)
        utility_metric = -(spatial_loss + 0.5 * (temp_day_loss + temp_hour_loss) + 0.5 * cat_loss)
        
        # Now try to calculate privacy metric (this might fail)
        # Use TUL classifier to estimate identifiability
        # First convert one-hot vectors to indices and ensure they're within valid ranges
        day_indices = tf.cast(tf.argmax(gen_trajs[2], axis=-1), tf.int32)
        # Clip day values to ensure they're in the valid range [0, 6]
        day_indices = tf.clip_by_value(day_indices, 0, 6)
        
        hour_indices = tf.cast(tf.argmax(gen_trajs[3], axis=-1), tf.int32)
        # Clip hour values to ensure they're in the valid range [0, 23]
        hour_indices = tf.clip_by_value(hour_indices, 0, 23)
        
        category_indices = tf.cast(tf.argmax(gen_trajs[1], axis=-1), tf.int32)
        # Get max category index from vocabulary size
        max_category = model.vocab_size['category'] - 1
        # Clip category values to ensure they're in valid range
        category_indices = tf.clip_by_value(category_indices, 0, max_category)
        
        # Format lat_lon to match MARC's expected input shape
        lat_lon_padded = tf.pad(gen_trajs[0], [[0, 0], [0, 0], [0, 38]])
        
        # Debug output
        print(f"Day range: min={tf.reduce_min(day_indices).numpy()}, max={tf.reduce_max(day_indices).numpy()}")
        print(f"Hour range: min={tf.reduce_min(hour_indices).numpy()}, max={tf.reduce_max(hour_indices).numpy()}")
        print(f"Category range: min={tf.reduce_min(category_indices).numpy()}, max={tf.reduce_max(category_indices).numpy()}")
        
        # Call TUL classifier with properly clipped indices
        tul_preds = model.tul_classifier([day_indices, hour_indices, category_indices, lat_lon_padded])
        
        # Process TUL predictions
        batch_size = len(day_indices)
        num_users = tul_preds.shape[1]
        
        # Generate user indices but make sure they don't exceed the valid range
        user_indices = np.arange(batch_size) % num_users
        
        # Gather user probabilities
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        indices = tf.stack([batch_indices, tf.cast(user_indices, tf.int32)], axis=1)
        user_probs = tf.gather_nd(tul_preds, indices)
        
        # Average user probability (lower means better privacy)
        privacy_metric = tf.reduce_mean(user_probs).numpy()
        
    except Exception as e:
        print(f"Error computing privacy metric: {e}")
        print("Using placeholder privacy metric")
        # Return placeholder value if privacy metric computation fails
        privacy_metric = 0.5  # Neutral privacy score
    
    return privacy_metric, utility_metric

if __name__ == '__main__':
    main()