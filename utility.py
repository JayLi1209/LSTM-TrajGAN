import pandas as pd
import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt

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
    
    # 1. Trajectory Count Preservation
    real_traj_count = real_df['tid'].nunique()
    syn_traj_count = syn_df['tid'].nunique()
    metrics['trajectory_count_ratio'] = syn_traj_count / real_traj_count
    
    # 2. Category Distribution Preservation
    real_category_dist = real_df['category'].value_counts(normalize=True)
    syn_category_dist = syn_df['category'].value_counts(normalize=True)
    
    # Handle missing categories
    all_categories = sorted(set(real_category_dist.index) | set(syn_category_dist.index))
    real_dist_array = np.array([real_category_dist.get(cat, 0) for cat in all_categories])
    syn_dist_array = np.array([syn_category_dist.get(cat, 0) for cat in all_categories])
    
    # Jensen-Shannon Divergence (symmetric version of KL divergence)
    metrics['category_js_divergence'] = calculate_js_divergence(real_dist_array, syn_dist_array)
    
    # 3. Temporal Distribution Preservation
    # Hour distribution
    real_hour_dist = real_df['hour'].value_counts(normalize=True)
    syn_hour_dist = syn_df['hour'].value_counts(normalize=True)
    
    all_hours = sorted(set(real_hour_dist.index) | set(syn_hour_dist.index))
    real_hour_array = np.array([real_hour_dist.get(h, 0) for h in all_hours])
    syn_hour_array = np.array([syn_hour_dist.get(h, 0) for h in all_hours])
    
    metrics['hour_js_divergence'] = calculate_js_divergence(real_hour_array, syn_hour_array)
    
    # Day distribution
    real_day_dist = real_df['day'].value_counts(normalize=True)
    syn_day_dist = syn_df['day'].value_counts(normalize=True)
    
    all_days = sorted(set(real_day_dist.index) | set(syn_day_dist.index))
    real_day_array = np.array([real_day_dist.get(d, 0) for d in all_days])
    syn_day_array = np.array([syn_day_dist.get(d, 0) for d in all_days])
    
    metrics['day_js_divergence'] = calculate_js_divergence(real_day_array, syn_day_array)
    
    # 4. Spatial Distribution Preservation
    # Compare overall spatial distribution using grid-based approach
    lat_min = min(real_df['lat'].min(), syn_df['lat'].min())
    lat_max = max(real_df['lat'].max(), syn_df['lat'].max())
    lon_min = min(real_df['lon'].min(), syn_df['lon'].min())
    lon_max = max(real_df['lon'].max(), syn_df['lon'].max())
    
    # Create grid cells
    grid_size = 0.01  # approximately 1km depending on latitude
    metrics['spatial_js_divergence'] = calculate_spatial_js_divergence(
        real_df, syn_df, lat_min, lat_max, lon_min, lon_max, grid_size)
    
    # 5. Trajectory-level Metrics
    common_tids = set(real_df['tid'].unique()) & set(syn_df['tid'].unique())
    
    if len(common_tids) > 0:
        # Calculate metrics for trajectories with same IDs
        traj_metrics = calculate_trajectory_specific_metrics(real_df, syn_df, common_tids)
        metrics.update(traj_metrics)
    
    # 6. Overall utility score (weighted combination of metrics)
    # Lower divergence values are better (closer to 0)
    # Convert divergences to similarities (1 - normalized_divergence)
    
    # Normalize JS divergences to 0-1 scale (they're already between 0-1, but invert for similarity)
    category_similarity = 1 - metrics['category_js_divergence']
    hour_similarity = 1 - metrics['hour_js_divergence']
    day_similarity = 1 - metrics['day_js_divergence']
    spatial_similarity = 1 - metrics['spatial_js_divergence']
    
    # Calculate overall utility score (equal weights for simplicity)
    metrics['overall_utility_score'] = 0.25 * category_similarity + \
                                     0.25 * hour_similarity + \
                                     0.25 * day_similarity + \
                                     0.25 * spatial_similarity
    
    return metrics

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
    synthetic_file = "results/syn_traj_test.csv"
    
    # Calculate metrics
    utility_metrics = evaluate_trajectory_utility(real_file, synthetic_file)
    
    # Display results
    print("Trajectory Utility Preservation Metrics:")
    for metric, value in utility_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results
    visualize_metrics(utility_metrics, "trajectory_utility_metrics.png")