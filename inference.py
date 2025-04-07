import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from model import RL_Enhanced_Transformer_TrajGAN, TransformerBlock
from tensorflow.keras.layers import Input, Dense, Lambda, TimeDistributed, Concatenate, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.models import Model
import json
import os

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

def generate_trajectories(model, test_data, num_samples=1000, batch_size=32):
    """Generate synthetic trajectories using the trained model"""
    generated_trajectories = []
    
    # Process test data in batches
    for i in range(0, len(test_data[0]), batch_size):
        batch = [data[i:i+batch_size] for data in test_data]
        
        # Generate noise for the batch
        noise = np.random.normal(0, 1, (len(batch[0]), model.latent_dim))
        
        # Generate trajectories
        gen_trajs = model.generator.predict([*batch, noise])
        
        # Store generated trajectories
        generated_trajectories.append(gen_trajs)
    
    # Concatenate all batches
    generated_trajectories = [np.concatenate([batch[i] for batch in generated_trajectories], axis=0)
                            for i in range(len(generated_trajectories[0]))]
    
    return generated_trajectories

def save_generated_trajectories(trajectories, output_path):
    """Save generated trajectories to CSV file"""
    # Convert trajectories to DataFrame format
    num_trajectories = len(trajectories[0])
    num_points = trajectories[0].shape[1]
    
    data = []
    for traj_idx in range(num_trajectories):
        for point_idx in range(num_points):
            if trajectories[4][traj_idx, point_idx, 0] > 0:  # Only include non-padded points
                data.append({
                    'tid': traj_idx,
                    'lat': trajectories[0][traj_idx, point_idx, 0],
                    'lon': trajectories[0][traj_idx, point_idx, 1],
                    'category': np.argmax(trajectories[1][traj_idx, point_idx]),
                    'day': np.argmax(trajectories[2][traj_idx, point_idx]),
                    'hour': np.argmax(trajectories[3][traj_idx, point_idx])
                })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated trajectories saved to {output_path}")

def main():
    # Model parameters
    vocab_size = {
        'lat_lon': 2,
        'day': 7,
        'hour': 24,
        'category': 10,
        'mask': 1,
    }
    
    # Load model configuration
    with open('results/model_config_100.json', 'r') as f:
        config = json.load(f)
    
    # Filter out non-initialization parameters and set embedding dimension to 100
    init_params = {
        'latent_dim': 100,  # Changed back to 100 to match training
        'keys': config['keys'],
        'vocab_size': config['vocab_size'],
        'max_length': config['max_length'],
        'lat_centroid': config['lat_centroid'],
        'lon_centroid': config['lon_centroid'],
        'scale_factor': config['scale_factor']
    }
    
    # Initialize model
    model = RL_Enhanced_Transformer_TrajGAN(**init_params)
    
    # Override the model's build methods to match the saved architecture
    def build_generator(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        # Noise input
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        
        # Embedding layers for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Add noise input to the inputs list
        inputs.append(noise)
        
        # Add noise embedding
        noise_repeated = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.max_length, 1]))(noise)
        embeddings.append(noise_repeated)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Output layers
        outputs = []
        for _, key in enumerate(self.keys):
            if key == 'mask':
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == 'lat_lon':
                output = TimeDistributed(Dense(2, activation='tanh'), name='output_latlon')(x)
                scale_factor = self.scale_factor
                output_stratched = Lambda(lambda x: x * scale_factor)(output)
                outputs.append(output_stratched)
            else:
                output = TimeDistributed(Dense(self.vocab_size[key], activation='softmax'), 
                                       name='output_' + key)(x)
                outputs.append(output)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def build_discriminator(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True,
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True,
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(64, activation='relu')(concat_input)
        
        # Single Transformer block
        x = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=128, rate=0.2)(concat_input, training=True)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Add dropout for regularization
        x = Dropout(0.3)(x)
        
        # Output with reduced activation slope
        sigmoid = Dense(1, activation='sigmoid', kernel_initializer=he_uniform(seed=1))(x)
        
        return Model(inputs=inputs, outputs=sigmoid)
    
    # Override the model's build methods
    model.build_generator = build_generator.__get__(model, RL_Enhanced_Transformer_TrajGAN)
    model.build_discriminator = build_discriminator.__get__(model, RL_Enhanced_Transformer_TrajGAN)
    
    # Rebuild the discriminator with 64-dimensional embeddings
    model.discriminator = model.build_discriminator()
    
    # Load model weights
    model.generator.load_weights('results/generator_100.weights.h5')
    model.discriminator.load_weights('results/discriminator_100.weights.h5')
    model.critic.load_weights('results/critic_100.weights.h5')
    
    # Load and process test data
    test_data = load_and_process_data('data/test_latlon.csv', model.max_length, vocab_size)
    
    # Generate synthetic trajectories
    generated_trajectories = generate_trajectories(model, test_data, num_samples=1000)
    
    # Save generated trajectories
    save_generated_trajectories(generated_trajectories, 'results/syn_traj_test.csv')

if __name__ == "__main__":
    main() 