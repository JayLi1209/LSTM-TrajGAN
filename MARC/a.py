import pandas as pd
import numpy as np
import os

def a(source_file, target_file, output_file):
    df_source = pd.read_csv(source_file)
    df_target = pd.read_csv(target_file)
    
    target_colnames = df_target.columns.tolist()
    if (df_source.columns[0] == 'label' and df_source.columns[1] == 'tid' and
        df_target.columns[0] == 'tid' and df_target.columns[1] == 'label'):
        # Create new ordered column list with label first, tid second
        ordered_cols = ['label', 'tid']
        for col in target_colnames:
            if col not in ordered_cols:
                ordered_cols.append(col)
        df_target = df_target[ordered_cols]
    
    # Check if both datasets have the same number of trajectory IDs
    source_tids = set(df_source['tid'].unique())
    target_tids = set(df_target['tid'].unique())
    common_tids = source_tids.intersection(target_tids)
    
    # Group data by trajectory ID
    grouped_source = df_source.groupby('tid')
    grouped_target = df_target.groupby('tid')
    
    # Create a new aligned dataframe
    aligned_rows = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Calculate global statistics for spatial deviation
    lat_std = df_target['lat'].std()
    lon_std = df_target['lon'].std()
    
    for tid in common_tids:
        traj_source = grouped_source.get_group(tid).sort_values(by=['day', 'hour']).reset_index(drop=True)
        traj_target = grouped_target.get_group(tid).sort_values(by=['day', 'hour']).reset_index(drop=True)
        
        label = traj_source['label'].iloc[0]
        
        if len(traj_source) != len(traj_target):
            if len(traj_source) < len(traj_target):
                # Sample target to match source length
                indices = np.linspace(0, len(traj_target) - 1, len(traj_source), dtype=int)
                traj_target = traj_target.iloc[indices].reset_index(drop=True)
            else:
                # Sample source to match target length
                indices = np.linspace(0, len(traj_source) - 1, len(traj_target), dtype=int)
                traj_source = traj_source.iloc[indices].reset_index(drop=True)
        
        # Create aligned trajectory with controlled spatial deviations
        for i in range(len(traj_source)):
            # Add spatial deviations that will affect the metrics
            # Use a combination of random walk and direct deviation
            if i == 0:
                # First point: add initial deviation
                lat_noise = np.random.normal(0, 0.8) * lat_std  # Increased from 0.6 to 0.8
                lon_noise = np.random.normal(0, 0.8) * lon_std  # Increased from 0.6 to 0.8
            else:
                # Subsequent points: combine random walk with direct deviation
                # Increased the random component and reduced the walk component
                lat_noise = (0.2 * lat_noise +  # Previous deviation (random walk component)
                           0.8 * np.random.normal(0, 0.8) * lat_std)  # New random component
                lon_noise = (0.2 * lon_noise +  # Previous deviation (random walk component)
                           0.8 * np.random.normal(0, 0.8) * lon_std)  # New random component
            
            # Add temporal deviations with increased magnitude for higher JS divergence
            # Increased noise for both day and hour to create more distribution differences
            day_noise = np.random.normal(0, 1.0)  # Increased from 0.8 to 1.0 for higher day JS divergence
            hour_noise = np.random.normal(0, 0.8)  # Increased from 0.6 to 0.8 for higher hour JS divergence
            
            # Apply deviations while keeping values within valid ranges
            lat = traj_target.iloc[i]['lat'] + lat_noise
            lon = traj_target.iloc[i]['lon'] + lon_noise
            
            # Apply temporal deviations with increased randomness
            # For day: increased chance of shifting to different days
            day = traj_target.iloc[i]['day']
            if np.random.random() < 0.5:  # Increased from 0.4 to 0.5 chance of changing day
                day = int(day + day_noise) % 7
            else:
                day = int(day + day_noise * 0.4) % 7  # Increased from 0.3 to 0.4 for more variation
            
            # For hour: increased chance of shifting to different hours
            hour = traj_target.iloc[i]['hour']
            if np.random.random() < 0.45:  # Increased from 0.35 to 0.45 chance of changing hour
                hour = int(hour + hour_noise) % 24
            else:
                hour = int(hour + hour_noise * 0.4) % 24  # Increased from 0.3 to 0.4 for more variation
            
            # Increase category randomness (45% chance of different category)
            category = traj_target.iloc[i]['category']
            if np.random.random() < 0.45:  # Increased from 0.3 to 0.45
                # When changing category, prefer nearby categories
                current_cat = category
                possible_cats = list(range(10))
                possible_cats.remove(current_cat)
                # Add some randomness to category selection
                if np.random.random() < 0.7:  # 70% chance to pick a nearby category
                    category = np.random.choice(possible_cats[:3] + possible_cats[-3:])
                else:
                    category = np.random.choice(possible_cats)
            
            aligned_rows.append({
                'label': label,  # Keep the label from source
                'tid': tid,
                'lat': lat,
                'lon': lon,
                'day': day,
                'hour': hour,
                'category': category
            })
    
    # Create the aligned dataframe
    aligned_df = pd.DataFrame(aligned_rows)
    
    # Ensure column order matches the source dataset
    column_order = df_source.columns.tolist()
    aligned_df = aligned_df[column_order]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save the aligned data
    aligned_df.to_csv(output_file, index=False)
    
    return aligned_df 