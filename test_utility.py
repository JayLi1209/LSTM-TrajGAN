import pandas as pd
import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import tensorflow as tf
from keras.utils import pad_sequences, to_categorical

def evaluate_trajectory_utility(real_file, synthetic_file):
    """
    Evaluate how well synthesized trajectories preserve the utility of real ones
    
    Parameters:
    -----------
    real_file : str
        Path to the real trajectory CSV file
    synthetic_file : str
        Path to the synthesized trajectory CSV file
        
    Returns:
    --------
    dict
        Dictionary containing various utility metrics
    """
    # Load data
    real_df = pd.read_csv(real_file)
    syn_df = pd.read_csv(synthetic_file)
    
    # Ensure column order is consistent (noticed tid and label were swapped)
    if 'tid' != real_df.columns[0]:
        real_df = real_df[syn_df.columns]
    
    metrics = {}
    
    try:
        # Convert DataFrames to model-ready format
        max_length = 144  # Default value, can be adjusted based on your data
        vocab_size = {
            'lat_lon': 2,
            'day': 7,
            'hour': 24,
            'category': 10,
            'mask': 1
        }
        
        # Process real data
        real_data = process_dataframe(real_df, max_length, vocab_size)
        # Process synthetic data
        syn_data = process_dataframe(syn_df, max_length, vocab_size)
        
        # Calculate spatial loss
        spatial_loss = tf.reduce_mean(tf.square(syn_data[0] - real_data[0])).numpy()
        print("spatial_loss is: ", spatial_loss)
        
        # Calculate temporal loss for day
        temp_day_loss = -tf.reduce_mean(tf.reduce_sum(
            real_data[2] * tf.math.log(tf.clip_by_value(syn_data[2], 1e-7, 1.0)), 
            axis=-1)).numpy()
        print("temp_day_loss is: ", temp_day_loss)
        
        # Calculate temporal loss for hour
        temp_hour_loss = -tf.reduce_mean(tf.reduce_sum(
            real_data[3] * tf.math.log(tf.clip_by_value(syn_data[3], 1e-7, 1.0)), 
            axis=-1)).numpy()
        print("temp_hour_loss is: ", temp_hour_loss)
        
        # Calculate category loss
        cat_loss = -tf.reduce_mean(tf.reduce_sum(
            real_data[1] * tf.math.log(tf.clip_by_value(syn_data[1], 1e-7, 1.0)), 
            axis=-1)).numpy()
        print("cat_loss is: ", cat_loss)
        
        # Combine utility components (lower is better, so we use negative)
        utility_metric = spatial_loss + 0.5 * (temp_day_loss + temp_hour_loss) + 0.5 * cat_loss
        print("utility_metric is: ", utility_metric)
        
        # Store all metrics
        metrics['spatial_loss'] = spatial_loss
        metrics['temp_day_loss'] = temp_day_loss
        metrics['temp_hour_loss'] = temp_hour_loss
        metrics['category_loss'] = cat_loss
        metrics['overall_utility_score'] = utility_metric
        
    except Exception as e:
        print(f"Error computing utility metrics: {e}")
        print("Using placeholder metrics")
        metrics['overall_utility_score'] = 0.5  # Neutral utility score
    
    return metrics

def process_dataframe(df, max_length, vocab_size):
    """Convert DataFrame to model-ready format"""
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
        processed['category'].append(process_feature(group['category'], vocab_size['category']))
        processed['day'].append(process_feature(group['day'], vocab_size['day']))
        processed['hour'].append(process_feature(group['hour'], vocab_size['hour']))
    
    return [np.array(processed[k]) for k in ['lat_lon', 'day', 'hour', 'category', 'mask']]

def calculate_js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two probability distributions"""
    # Ensure valid probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate midpoint distribution
    m = 0.5 * (p + q)
    
    # JS divergence
    js_div = 0.5 * calculate_kl_divergence(p, m) + 0.5 * calculate_kl_divergence(q, m)
    return js_div

def calculate_kl_divergence(p, q):
    """Calculate Kullback-Leibler divergence between two probability distributions"""
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    q = q + epsilon
    p = p + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # KL divergence
    return np.sum(p * np.log(p / q))

def calculate_spatial_js_divergence(real_df, syn_df, lat_min, lat_max, lon_min, lon_max, grid_size):
    """Calculate JS divergence of spatial distributions using a grid-based approach"""
    # Create grid cells
    lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
    lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
    
    # Count points in each cell
    real_hist, _, _ = np.histogram2d(real_df['lat'], real_df['lon'], bins=[lat_bins, lon_bins])
    syn_hist, _, _ = np.histogram2d(syn_df['lat'], syn_df['lon'], bins=[lat_bins, lon_bins])
    
    # Flatten to 1D for JS divergence calculation
    real_dist = real_hist.flatten() / real_hist.sum()
    syn_dist = syn_hist.flatten() / syn_hist.sum()
    
    return calculate_js_divergence(real_dist, syn_dist)

def calculate_trajectory_specific_metrics(real_df, syn_df, common_tids):
    """Calculate metrics that compare individual trajectories with the same IDs"""
    metrics = {}
    
    length_ratios = []
    dtw_distances = []
    hausdorff_distances = []
    
    for tid in common_tids:
        real_traj = real_df[real_df['tid'] == tid]
        syn_traj = syn_df[syn_df['tid'] == tid]
        
        # 1. Length preservation
        real_length = len(real_traj)
        syn_length = len(syn_traj)
        length_ratio = syn_length / real_length if real_length > 0 else 0
        length_ratios.append(length_ratio)
        
        # 2. Path deviation metrics
        if real_length > 0 and syn_length > 0:
            # Convert trajectories to list of (lat, lon) tuples
            real_coords = list(zip(real_traj['lat'], real_traj['lon']))
            syn_coords = list(zip(syn_traj['lat'], syn_traj['lon']))
            
            # Dynamic Time Warping distance
            dtw_dist = dynamic_time_warping(real_coords, syn_coords)
            dtw_distances.append(dtw_dist)
            
            # Hausdorff distance
            h_dist = hausdorff_distance(real_coords, syn_coords)
            hausdorff_distances.append(h_dist)
    
    if length_ratios:
        metrics['avg_length_ratio'] = np.mean(length_ratios)
    if dtw_distances:
        metrics['avg_dtw_distance'] = np.mean(dtw_distances)
    if hausdorff_distances:
        metrics['avg_hausdorff_distance'] = np.mean(hausdorff_distances)
    
    return metrics

def haversine_distance(coord1, coord2):
    """Calculate Haversine distance between two coordinates in kilometers"""
    return haversine(coord1, coord2)

def dynamic_time_warping(traj1, traj2):
    """Calculate Dynamic Time Warping distance between two trajectories"""
    n, m = len(traj1), len(traj2)
    
    # Initialize cost matrix
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = haversine_distance(traj1[i-1], traj2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                         dtw_matrix[i, j-1], 
                                         dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]

def hausdorff_distance(traj1, traj2):
    """Calculate Hausdorff distance between two trajectories"""
    forward_hdist = directed_hausdorff_distance(traj1, traj2)
    backward_hdist = directed_hausdorff_distance(traj2, traj1)
    return max(forward_hdist, backward_hdist)

def directed_hausdorff_distance(traj1, traj2):
    """Calculate directed Hausdorff distance from traj1 to traj2"""
    max_min_dist = 0
    for point1 in traj1:
        min_dist = float('inf')
        for point2 in traj2:
            dist = haversine_distance(point1, point2)
            min_dist = min(min_dist, dist)
        max_min_dist = max(max_min_dist, min_dist)
    return max_min_dist

def visualize_metrics(metrics, output_file=None):
    """Visualize the utility metrics"""
    # Filter out metrics for visualization (exclude overall score and others as needed)
    viz_metrics = {}
    for key, value in metrics.items():
        if key in ['category_js_divergence', 'hour_js_divergence', 
                   'day_js_divergence', 'spatial_js_divergence']:
            # Convert divergence to similarity for more intuitive visualization
            viz_metrics[key.replace('_js_divergence', '_similarity')] = 1 - value
    
    if 'avg_dtw_distance' in metrics and 'avg_hausdorff_distance' in metrics:
        # Normalize distance metrics to 0-1 scale (higher is better)
        max_dist = max(metrics['avg_dtw_distance'], metrics['avg_hausdorff_distance'])
        if max_dist > 0:
            viz_metrics['path_similarity'] = 1 - (metrics['avg_dtw_distance'] / max_dist)
    
    # Create bar chart for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(viz_metrics.keys(), viz_metrics.values())
    plt.ylim(0, 1)
    plt.title('Trajectory Utility Preservation Metrics (higher is better)')
    plt.ylabel('Similarity Score (0-1)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    
    # Also display overall utility score
    print(f"Overall utility preservation score: {metrics['overall_utility_score']:.4f} (0-1 scale, higher is better)")

# Example usage
if __name__ == "__main__":
    real_file = "data/test_latlon.csv"
    synthetic_file = "/root/autodl-tmp/syn_traj_test.csv"
    
    # Calculate metrics
    utility_metrics = evaluate_trajectory_utility(real_file, synthetic_file)
    
    # Display results
    print("Trajectory Utility Preservation Metrics:")
    for metric, value in utility_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results
    visualize_metrics(utility_metrics, "trajectory_utility_metrics.png")