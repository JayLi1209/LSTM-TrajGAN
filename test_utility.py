import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import RL_Enhanced_Transformer_TrajGAN
from MARC.marc import MARC

def load_and_process_data(csv_path, max_length, vocab_sizes):
    """Load trajectory data from CSV and process into model-ready format"""
    df = pd.read_csv(csv_path)
    trajectories = df.groupby('tid')
    
    processed = {
        'lat_lon': [],
        'day': [],
        'hour': [],
        'category': [],
        'mask': [],
    }
    
    for tid, group in trajectories:
        # Original sequence length
        seq_len = len(group)
        
        # Process lat/lon coordinates
        lat_lon = group[['lat', 'lon']].values.astype('float32')
        padded_ll = pad_sequences([lat_lon], maxlen=max_length, padding='pre', 
                                truncating='post', dtype='float32')[0]
        
        # Process categorical features with one-hot encoding
        def process_feature(values, vocab_size):
            padded = pad_sequences([values], maxlen=max_length, padding='pre',
                                truncating='post', dtype='int32')[0]
            return to_categorical(padded, num_classes=vocab_size)
        
        # Create mask (1 for real points, 0 for padding)
        mask = np.zeros(max_length, dtype='float32')
        mask[:seq_len] = 1.0
        mask = np.expand_dims(mask, axis=-1)
        
        # Store processed features
        processed['mask'].append(mask)
        processed['lat_lon'].append(padded_ll)
        processed['category'].append(process_feature(group['category'], vocab_sizes['category']))
        processed['day'].append(process_feature(group['day'], vocab_sizes['day']))
        processed['hour'].append(process_feature(group['hour'], vocab_sizes['hour']))

    return [np.array(processed[k]) for k in ['lat_lon', 'day', 'hour', 'category', 'mask']]

def compute_utility_metrics(real_data, gen_data):
    """Compute utility metrics between real and generated data"""
    # Spatial loss - lower is better
    spatial_loss = tf.reduce_mean(tf.square(gen_data[0] - real_data[0])).numpy()
    print("Spatial Loss:", spatial_loss)
    
    # Temporal loss - lower is better
    temp_day_loss = -tf.reduce_mean(tf.reduce_sum(
        real_data[2] * tf.math.log(tf.clip_by_value(gen_data[2], 1e-7, 1.0)), 
        axis=-1)).numpy()
    print("Temporal Day Loss:", temp_day_loss)
    
    temp_hour_loss = -tf.reduce_mean(tf.reduce_sum(
        real_data[3] * tf.math.log(tf.clip_by_value(gen_data[3], 1e-7, 1.0)), 
        axis=-1)).numpy()
    print("Temporal Hour Loss:", temp_hour_loss)
    
    # Category loss - lower is better
    cat_loss = -tf.reduce_mean(tf.reduce_sum(
        real_data[1] * tf.math.log(tf.clip_by_value(gen_data[1], 1e-7, 1.0)), 
        axis=-1)).numpy()
    print("Category Loss:", cat_loss)
    
    # Combine utility components (lower is better)
    utility_metric = spatial_loss + 0.5 * (temp_day_loss + temp_hour_loss) + 0.5 * cat_loss
    print("Overall Utility Metric:", utility_metric)
    
    return {
        'spatial_loss': spatial_loss,
        'temp_day_loss': temp_day_loss,
        'temp_hour_loss': temp_hour_loss,
        'cat_loss': cat_loss,
        'utility_metric': utility_metric
    }

def main():
    # Load data statistics
    data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    max_length = data_stats['max_length']
    
    # Define vocabulary sizes
    vocab_size = {
        'lat_lon': 2,
        'day': 7,
        'hour': 24,
        'category': 10,
        'mask': 1
    }
    
    # Load real test data
    print("Loading real test data...")
    real_data = load_and_process_data('data/test_latlon.csv', max_length, vocab_size)
    
    # Load generated data
    print("Loading generated data...")
    gen_data = load_and_process_data('results/generated_trajectories.csv', max_length, vocab_size)
    
    # Compute and compare utility metrics
    print("\nComputing utility metrics...")
    metrics = compute_utility_metrics(real_data, gen_data)
    
    # Save metrics to file
    np.save('results/utility_metrics.npy', metrics)
    print("\nMetrics saved to results/utility_metrics.npy")

if __name__ == '__main__':
    main()
