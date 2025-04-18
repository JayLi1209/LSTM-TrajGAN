import tensorflow as tf
import keras
import numpy as np
import random
from tensorflow.keras import layers
import os
import json

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from keras.initializers import he_uniform
from keras.regularizers import l1

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import CustomTrajLoss, compute_advantage, compute_returns

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class RL_Enhanced_Transformer_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # RL parameters
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.c1 = 0.5  # value function coefficient (reduced)
        self.c2 = 0.01  # entropy coefficient
        self.ppo_epochs = 4  # Number of PPO epochs per batch

        self.r_adv_ema = tf.Variable(0., trainable=False)  # EMA for reward components
        self.r_util_ema = tf.Variable(0., trainable=False)
        self.r_priv_ema = tf.Variable(0., trainable=False)
        self.ema_alpha = 0.95  # Smoothing factor
        
        # Load or initialize TUL classifier for privacy rewards
        self.tul_classifier = self.load_tul_classifier()
        
        # Define reward weights
        self.w_adv = 0.5  # Weight for adversarial reward (reduced)
        self.w_util = 0.5  # Weight for utility reward (reduced)
        self.w_priv = 0.5  # Weight for privacy reward (reduced)
        
        # Define utility component weights
        self.beta = 0.5   # Spatial loss weight (reduced)
        self.gamma = 0.2  # Temporal loss weight (reduced)
        self.chi = 0.2    # Category loss weight (reduced)
        self.alpha = 0.5  # Privacy strength weight (reduced)
        
        # Define optimizers with reduced learning rates and gradient clipping
        self.actor_optimizer = Adam(0.00001, clipnorm=1.0)  # Reduced from 0.00005
        self.critic_optimizer = Adam(0.00001, clipnorm=1.0)  # Reduced from 0.00005
        self.discriminator_optimizer = Adam(0.000005, clipnorm=1.0)  # Reduced from 0.00001

        # Build networks
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.discriminator = self.build_discriminator()
        
        # Compile models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
        self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
        
        # Combined model for training
        self.setup_combined_model()

    def get_config(self):
        """Return the configuration of the model for serialization."""
        return {
            "latent_dim": self.latent_dim,
            "keys": self.keys,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "lat_centroid": self.lat_centroid,
            "lon_centroid": self.lon_centroid,
            "scale_factor": self.scale_factor,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "c1": self.c1,
            "c2": self.c2,
        }
        
    @classmethod
    def from_config(cls, config):
        """Create a model from its config."""
        return cls(**config)

    @classmethod
    def from_saved_checkpoint(cls, epoch, checkpoint_dir='results'):
        """Load a model from saved checkpoints.
        
        Args:
            epoch: The epoch number of the checkpoint to load
            checkpoint_dir: Directory where checkpoints are saved
            
        Returns:
            An instance of the model with loaded weights
        """
        # Load model config
        try:
            with open(f'{checkpoint_dir}/model_config_{epoch}.json', 'r') as f:
                config = json.load(f)
            
            # Create model instance
            model = cls(**config)
            
            # Try to load saved weights
            model.generator.load_weights(f'{checkpoint_dir}/generator_{epoch}.weights.h5')
            model.discriminator.load_weights(f'{checkpoint_dir}/discriminator_{epoch}.weights.h5')
            model.critic.load_weights(f'{checkpoint_dir}/critic_{epoch}.weights.h5')
            
            print(f"Successfully loaded model from epoch {epoch}")
            return model
            
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            raise

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
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
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
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
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

    def build_critic(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        for _, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Value head
        value = Dense(1)(x)
        
        return Model(inputs=inputs, outputs=value)

    def build_discriminator(self):
        # Similar to original discriminator but with Transformer blocks
        inputs = []
        embeddings = []
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output
        sigmoid = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=sigmoid)

    def setup_combined_model(self):
        # Generator inputs
        inputs = []
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        
        # Create inputs in the same order as build_generator
        for _, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
            inputs.append(i)
        
        # Add noise as the last input (to match the generator's input order)
        inputs.append(noise)
        
        # Generate trajectories
        gen_trajs = self.generator(inputs)
        
        # Discriminator predictions
        pred = self.discriminator(gen_trajs[:4])
        
        # Create the combined model
        self.combined = Model(inputs, pred)
        
        # Create a custom loss instance for trajectory optimization
        self.traj_loss = CustomTrajLoss()
        # Store input and generator output references
        self.input_tensors = inputs
        self.generated_trajectories = gen_trajs
        
        # Compile the model with the custom loss
        self.combined.compile(loss=self.traj_loss, optimizer=self.actor_optimizer)
        
        # Store the generator outputs for later use in reward computation
        self.gen_trajs_symbolic = gen_trajs

    def compute_rewards(self, real_trajs, gen_trajs, tul_classifier):
        """Simplified reward calculation with core components preserved."""
        # Adversarial reward
        d_pred = tf.clip_by_value(self.discriminator(gen_trajs[:4]), 1e-7, 1-1e-7)
        r_adv = tf.math.log(d_pred)

        # Utility reward components
        spatial_loss = tf.clip_by_value(
            tf.reduce_mean(tf.square(gen_trajs[0] - real_trajs[0]), [1,2]), 0, 10)
        
        def cross_entropy(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
            return -tf.reduce_sum(y_true * tf.math.log(y_pred), [1,2])
        
        temp_loss = cross_entropy(real_trajs[2], gen_trajs[2]) + \
                    cross_entropy(real_trajs[3], gen_trajs[3])
        cat_loss = cross_entropy(real_trajs[1], gen_trajs[1])
        
        r_util = -(self.beta*spatial_loss + self.gamma*temp_loss + self.chi*cat_loss)

        # Privacy reward (simplified error handling)
        try:
            day = tf.argmax(gen_trajs[2], -1)
            hour = tf.argmax(gen_trajs[3], -1)
            category = tf.argmax(gen_trajs[1], -1)
            lat_lon = tf.pad(gen_trajs[0], [[0,0],[0,0],[0,38]])
            
            tul_preds = tul_classifier([day, hour, category, lat_lon])
            user_probs = tf.gather_nd(tul_preds, 
                tf.stack([tf.range(tf.shape(day)[0]), 
                tf.argmax(tul_preds, -1)], 1))
            r_priv = -self.alpha * user_probs
        except Exception as e:
            r_priv = tf.zeros_like(r_adv)

        # Combined reward
        rewards = (self.w_adv*r_adv + self.w_util*r_util + self.w_priv*r_priv)
        rewards = (rewards - tf.reduce_mean(rewards)) / (tf.math.reduce_std(rewards) + 1e-8)

        def ema_norm(new_value, ema_var):
            new_ema = self.ema_alpha * ema_var + (1 - self.ema_alpha) * tf.reduce_mean(new_value)
            ema_var.assign(new_ema)
            return new_value / (tf.math.abs(new_ema) + 1e-8)

        r_adv = ema_norm(r_adv, self.r_adv_ema)
        r_util = ema_norm(r_util, self.r_util_ema) 
        r_priv = ema_norm(r_priv, self.r_priv_ema)

        # Use sigmoid transformation for bounded rewards
        combined_rewards = (
            tf.sigmoid(0.3 * r_adv) +
            tf.sigmoid(0.5 * r_util) + 
            tf.sigmoid(0.7 * r_priv)
        )
        
        return tf.clip_by_value(combined_rewards, 0.1, 0.9)

    def train_step(self, real_trajs, batch_size=256):
        # Generate trajectories
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        print("Shapes of real_trajs elements:")
        for i, tensor in enumerate(real_trajs):
            print(f"real_trajs[{i}]: {tensor.shape}")
        gen_trajs = self.generator.predict([*real_trajs, noise])
        
        # Ensure consistent data types
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        
        # Update the custom loss with the current real and generated trajectories
        self.traj_loss.set_trajectories(real_trajs, gen_trajs)
        
        # Compute full rewards using the TUL classifier
        rewards = self.compute_rewards(real_trajs, gen_trajs, self.tul_classifier)
        
        # Compute advantages and returns for PPO
        values = self.critic.predict(real_trajs[:4])
        values = tf.cast(values, tf.float32)
        advantages = compute_advantage(rewards, values, self.gamma, self.gae_lambda)
        returns = compute_returns(rewards, self.gamma)
        
        # Ensure returns has the same batch size as what the critic expects
        # The critic takes real_trajs[:4] as input, so returns should match that shape
        if returns.shape[0] != batch_size:
            print(f"Warning: Reshaping returns from {returns.shape} to [{batch_size}, 1]")
            returns = tf.reshape(returns, [batch_size, 1])
        
        # Update critic using returns
        c_loss = self.critic.train_on_batch(real_trajs[:4], returns)
        
        # Update discriminator
        d_loss_real = self.discriminator.train_on_batch(
            real_trajs[:4],
            np.ones((batch_size, 1))
        )
        d_loss_fake = self.discriminator.train_on_batch(
            gen_trajs[:4],
            np.zeros((batch_size, 1))
        )
        
        # Update generator (actor) using advantages
        g_loss = self.update_actor(real_trajs, gen_trajs, advantages)
        
        return {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake, "g_loss": g_loss, "c_loss": c_loss}

    def train(self, epochs=1, batch_size=256, sample_interval=10):
        # Training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        self.x_train = x_train
        
        # Padding
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') 
                  for f in x_train]
        self.X_train = X_train
        
        # Check if we need to rebuild the model with correct input shapes
        needs_rebuild = False
        actual_shapes = {}
        
        for i, key in enumerate(self.keys):
            if key != 'mask':
                actual_shape = X_train[i].shape
                print(f"Data shape for {key}: {actual_shape}")
                if key == 'category' and actual_shape[2] != self.vocab_size[key]:
                    print(f"Mismatch for {key}: expected {self.vocab_size[key]}, got {actual_shape[2]}")
                    self.vocab_size[key] = actual_shape[2]
                    needs_rebuild = True
                actual_shapes[key] = actual_shape[2]
        
        # Rebuild the model if needed
        if needs_rebuild:
            print("Rebuilding model with correct input shapes...")
            # Save optimizer state
            optimizer_weights = None
            if hasattr(self, 'actor_optimizer') and hasattr(self.actor_optimizer, 'get_weights'):
                optimizer_weights = self.actor_optimizer.get_weights()
            
            # Rebuild models
            self.generator = self.build_generator()
            self.critic = self.build_critic()
            self.discriminator = self.build_discriminator()
            
            # Compile models
            self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
            self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
            
            self.setup_combined_model()
            
            # Restore optimizer state if available
            if optimizer_weights is not None:
                self.actor_optimizer.set_weights(optimizer_weights)
            
            print("Model rebuilt successfully!")
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            # Sample batch
            idx = np.random.randint(0, len(X_train[0]), batch_size)
            batch = [X[idx] for X in X_train]
            
            # Training step
            metrics = self.train_step(batch, batch_size)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"D_real: {metrics['d_loss_real']:.4f}, D_fake: {metrics['d_loss_fake']:.4f}, G: {metrics['g_loss']:.4f}, C: {metrics['c_loss']:.4f}")
            
            # Save checkpoints
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        # Make sure the results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save model weights
        try:
            self.generator.save_weights(f'results/generator_{epoch}.weights.h5')
            self.discriminator.save_weights(f'results/discriminator_{epoch}.weights.h5')
            self.critic.save_weights(f'results/critic_{epoch}.weights.h5')
            print(f"Model weights saved for epoch {epoch}")
        except Exception as e:
            print(f"Warning: Could not save weights for epoch {epoch}: {e}")
        
        # Now try to save the full models with architecture
        try:
            # Save the Keras models
            self.generator.save(f'results/generator_architecture_{epoch}.keras')
            self.discriminator.save(f'results/discriminator_architecture_{epoch}.keras')
            self.critic.save(f'results/critic_architecture_{epoch}.keras')
            print(f"Model architectures saved for epoch {epoch}")
            
            # Also save the main model's configuration
            with open(f'results/model_config_{epoch}.json', 'w') as f:
                json.dump(self.get_config(), f, indent=4)
            
        except Exception as e:
            print(f"Warning: Could not save full model architectures for epoch {epoch}: {e}")
            print("Only weights were saved. You'll need to recreate the model structure to load them.")

    def update_actor(self, states, actions, advantages):
        """Update generator using a simplified policy gradient approach."""
        # Get a single batch of data with random noise for the generator
        batch_size = states[0].shape[0]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Prepare inputs for the generator
        all_inputs = [*states, noise]
        
        # Use ones as targets for the discriminator output (we want to fool the discriminator)
        targets = np.ones((batch_size, 1))
        
        # Convert advantages to numpy and ensure proper type/shape
        try:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
            advantages = tf.tanh(advantages)  # Constrain to [-1, 1]
            
            # Use rank-based weighting instead of raw values
            advantages_np = tf.argsort(tf.argsort(advantages)).numpy()
            advantages_np = advantages_np / np.max(advantages_np)
            
            # Normalize to reasonable values (0 to 1 range)
            if np.max(advantages_np) > 0:
                advantages_np = advantages_np / np.max(advantages_np)
                
            # Ensure it has the right shape
            if len(advantages_np) != batch_size:
                print(f"Warning: Reshaping advantages from {len(advantages_np)} to {batch_size}")
                # If shapes don't match, use uniform weights
                advantages_np = np.ones(batch_size)
                
            # Final safety check - replace any NaN or inf values
            advantages_np = np.nan_to_num(advantages_np, nan=0.5, posinf=1.0, neginf=0.0)
                
        except Exception as e:
            print(f"Error processing advantages: {e}")
            # Fallback to uniform weights
            advantages_np = np.ones(batch_size)
            
        # Print statistics about advantage values used for training
        print(f"Advantage stats - min: {np.min(advantages_np):.4f}, max: {np.max(advantages_np):.4f}, " +
              f"mean: {np.mean(advantages_np):.4f}, std: {np.std(advantages_np):.4f}")
        
        # Train the combined model with sample weights from advantages
        loss = self.combined.train_on_batch(
            all_inputs, 
            targets,
            sample_weight=advantages_np
        )
        
        # If loss is extremely high, clip it for reporting purposes
        if loss > 10000:  # Arbitrary threshold for "too high" loss
            print(f"Warning: Loss is very high ({loss:.2f}), consider reducing learning rate manually")
            loss = min(loss, 10000.0)
        
        return loss

    def compute_trajectory_ratio(self, new_predictions, old_predictions):
        """Compute the PPO policy ratio between new and old policies for trajectory data.
        
        For trajectories, we take the product of point-wise prediction ratios,
        which is equivalent to the ratio of trajectory probabilities under each policy.
        """
        # Initialize ratio as ones
        ratio = tf.ones(shape=(len(new_predictions[0]), 1))
        
        # For each categorical output (category, day, hour), compute ratio
        for i in range(1, 4):  # Categorical outputs
            min_prob = 1e-5
            new_probs = tf.clip_by_value(new_probs, min_prob, 1.0-min_prob)
            old_probs = tf.clip_by_value(old_probs, min_prob, 1.0-min_prob)
            
            # Use log-space calculation for numerical stability
            log_ratio = tf.reduce_sum(
                tf.math.log(new_probs) - tf.math.log(old_probs),
                axis=[1,2]  # Sum over sequence and categorical dimensions
            )
        
        return tf.exp(tf.clip_by_value(log_ratio, -0.2, 0.2))
        
        # # For continuous outputs (coordinates), use normal distribution likelihood ratio
        # # This is simplified - in practice you'd need proper distribution modeling
        # coord_ratio = tf.exp(-0.5 * tf.reduce_sum(tf.square(new_predictions[0] - old_predictions[0]), axis=-1, keepdims=True))
        # ratio = ratio * coord_ratio
        
        # # Mask out padding
        # mask = new_predictions[4]
        # ratio = ratio * mask
        
        # # Average over the trajectory length
        # ratio = tf.reduce_mean(ratio, axis=1)
        
        # return ratio

    def load_tul_classifier(self):
        """Load the pre-trained Trajectory-User Linking classifier.
        
        Returns:
            A trained model that predicts user IDs from trajectories.
        """
        try:
            # Import MARC class and initialize it
            from MARC.marc import MARC
            
            # Create and initialize the MARC model
            marc_model = MARC()
            marc_model.build_model()
            
            # Load pre-trained weights
            marc_model.load_weights('./MARC/MARC_Weight.h5')
            
            print("Loaded pre-trained MARC TUL classifier")
            return marc_model
            
        except Exception as e:
            print(f"Error loading MARC model: {e}")
            print("Creating a fallback TUL classifier model")
            
            # If MARC model loading fails, create a fallback model that matches MARC's input format
            # MARC expects 4 inputs: day, hour, category (all as indices), and lat_lon (with shape 144,40)
            
            # Create input layers with the same names and shapes as MARC
            input_day = Input(shape=(144,), dtype='int32', name='input_day')
            input_hour = Input(shape=(144,), dtype='int32', name='input_hour')
            input_category = Input(shape=(144,), dtype='int32', name='input_category')
            input_lat_lon = Input(shape=(144, 40), name='input_lat_lon')
            
            # Create embeddings like MARC
            emb_day = Embedding(input_dim=7, output_dim=32, input_length=144)(input_day)
            emb_hour = Embedding(input_dim=24, output_dim=32, input_length=144)(input_hour)
            emb_category = Embedding(input_dim=10, output_dim=32, input_length=144)(input_category)
            
            # Process lat_lon
            lat_lon_dense = Dense(32, activation='relu')(input_lat_lon)
            
            # Concatenate all embeddings
            concat = Concatenate(axis=2)([emb_day, emb_hour, emb_category, lat_lon_dense])
            
            # LSTM layer for sequence processing
            lstm_out = LSTM(64, return_sequences=False)(concat)
            
            # Dense layers
            dense1 = Dense(128, activation='relu')(lstm_out)
            
            # Output layer - assuming 100 users for classification
            # (We'll adjust this if needed based on the actual dataset)
            output = Dense(193, activation='softmax')(dense1)
            
            # Create model
            fallback_model = Model(
                inputs=[input_day, input_hour, input_category, input_lat_lon],
                outputs=output
            )
            
            fallback_model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy']
            )
            
            print("Created fallback TUL classifier model with matching input format")
            
            # Since the fallback model uses Keras functional API, it already supports __call__
            return fallback_model