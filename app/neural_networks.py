"""
Deep Learning and Neural Network Module
Advanced neural network architectures for environmental intelligence
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class EnvironmentalDataset(Dataset):
    """PyTorch Dataset for environmental data"""
    
    def __init__(self, features, targets, sequence_length=24):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length - 1]
        )

class LSTMPredictor(nn.Module):
    """LSTM-based predictor for time series forecasting"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Layer normalization and residual connection
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Use the last output for prediction
        final_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc_layers(final_output)
        
        return prediction

class CNNTimeSeriesClassifier(nn.Module):
    """CNN-based classifier for environmental anomaly detection"""
    
    def __init__(self, input_channels, sequence_length, num_classes, dropout=0.3):
        super(CNNTimeSeriesClassifier, self).__init__()
        
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 32, kernel_size=3, dropout=dropout),
            self._make_conv_block(32, 64, kernel_size=3, dropout=dropout),
            self._make_conv_block(64, 128, kernel_size=3, dropout=dropout),
            self._make_conv_block(128, 256, kernel_size=3, dropout=dropout)
        ])
        
        # Calculate the size after convolutions
        conv_output_size = self._calculate_conv_output_size(sequence_length)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),  # *2 for avg and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
    
    def _calculate_conv_output_size(self, input_size):
        # After 4 max pooling operations with kernel_size=2
        return input_size // (2 ** 4)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # Change to (batch_size, features, sequence_length)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        # Concatenate pooled features
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output

class TransformerPredictor(nn.Module):
    """Transformer-based model for environmental forecasting"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, src, src_mask=None):
        # Input projection
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src, src_mask)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # Use the last token for prediction
        final_output = output[:, -1, :]
        
        # Prediction
        prediction = self.predictor(final_output)
        
        return prediction

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)

class WaterQualityAutoencoder:
    """Autoencoder for water quality anomaly detection"""
    
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def train(self, X, epochs=100, batch_size=32, validation_split=0.2):
        """Train the autoencoder"""
        X_scaled = self.scaler.fit_transform(X)
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def detect_anomalies(self, X, threshold_percentile=95):
        """Detect anomalies using reconstruction error"""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(mse, threshold_percentile)
        
        anomalies = mse > threshold
        
        return {
            'anomalies': anomalies,
            'reconstruction_errors': mse,
            'threshold': threshold,
            'anomaly_scores': mse / threshold  # Normalized scores
        }

class EnvironmentalGAN:
    """Generative Adversarial Network for environmental data synthesis"""
    
    def __init__(self, input_dim, latent_dim=100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.combined = self._build_combined()
        
    def _build_generator(self):
        model = models.Sequential([
            layers.Dense(128, input_dim=self.latent_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.input_dim, activation='tanh'),
        ])
        
        noise = layers.Input(shape=(self.latent_dim,))
        generated_data = model(noise)
        
        return models.Model(noise, generated_data)
    
    def _build_discriminator(self):
        model = models.Sequential([
            layers.Dense(512, input_dim=self.input_dim, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid'),
        ])
        
        data = layers.Input(shape=(self.input_dim,))
        validity = model(data)
        
        discriminator = models.Model(data, validity)
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(0.0002, 0.5),
            metrics=['accuracy']
        )
        
        return discriminator
    
    def _build_combined(self):
        self.discriminator.trainable = False
        
        noise = layers.Input(shape=(self.latent_dim,))
        generated_data = self.generator(noise)
        validity = self.discriminator(generated_data)
        
        combined = models.Model(noise, validity)
        combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(0.0002, 0.5)
        )
        
        return combined
    
    def train(self, X, epochs=10000, batch_size=32, sample_interval=1000):
        """Train the GAN"""
        X_scaled = (X - X.mean()) / X.std()  # Normalize to roughly [-1, 1]
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_scaled.shape[0], batch_size)
            real_data = X_scaled[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}, D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")
        
        return {'d_losses': d_losses, 'g_losses': g_losses}
    
    def generate_synthetic_data(self, n_samples):
        """Generate synthetic environmental data"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        synthetic_data = self.generator.predict(noise)
        return synthetic_data

class DeepQLearningAgent:
    """Deep Q-Learning agent for environmental control optimization"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.update_target_network()
        
    def _build_q_network(self):
        model = models.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])
        
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        target_q_values_current = self.q_network.predict(states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values_current[i][actions[i]] = rewards[i]
            else:
                target_q_values_current[i][actions[i]] = rewards[i] + 0.95 * max_target_q_values[i]
        
        self.q_network.fit(states, target_q_values_current, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess_environmental_data(data, sequence_length=24):
    """Preprocess environmental data for neural network training"""
    # Handle missing values
    data_filled = data.fillna(data.mean())
    
    # Apply smoothing filter
    for column in data_filled.select_dtypes(include=[np.number]).columns:
        if len(data_filled) > 10:
            # Apply Butterworth filter for smoothing
            b, a = butter(3, 0.1, btype='low')
            data_filled[column] = filtfilt(b, a, data_filled[column])
    
    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_filled.select_dtypes(include=[np.number]))
    
    # Create sequences for time series models
    sequences = []
    targets = []
    
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        targets.append(scaled_data[i + sequence_length])
    
    return np.array(sequences), np.array(targets), scaler

def train_pytorch_lstm_model(X, y, input_size, hidden_size=64, num_epochs=100):
    """Train PyTorch LSTM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = EnvironmentalDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = LSTMPredictor(input_size=input_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return model, losses

def create_comprehensive_neural_network_suite():
    """Create and demonstrate a comprehensive neural network suite"""
    print("🧠 Creating Comprehensive Neural Network Suite...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # Simulate environmental data
    data = pd.DataFrame({
        'ph': np.random.normal(7.2, 0.5, n_samples),
        'dissolved_oxygen': np.random.normal(8.0, 1.2, n_samples),
        'temperature': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 1, n_samples),
        'turbidity': np.random.exponential(3, n_samples),
        'conductivity': np.random.normal(500, 100, n_samples),
        'nitrates': np.random.normal(2.0, 0.5, n_samples),
        'phosphates': np.random.normal(0.1, 0.05, n_samples),
        'flow_rate': np.random.normal(50, 10, n_samples)
    })
    
    print(f"📊 Generated {len(data)} samples with {len(data.columns)} features")
    
    # Preprocess data
    sequences, targets, scaler = preprocess_environmental_data(data, sequence_length=24)
    print(f"🔧 Preprocessed data: {sequences.shape} sequences, {targets.shape} targets")
    
    # 1. Train Autoencoder for anomaly detection
    print("\n🎯 Training Autoencoder for anomaly detection...")
    autoencoder = WaterQualityAutoencoder(input_dim=data.shape[1], encoding_dim=4)
    autoencoder_history = autoencoder.train(data.values, epochs=50, batch_size=32)
    
    # Detect anomalies
    anomaly_results = autoencoder.detect_anomalies(data.values)
    print(f"🚨 Detected {np.sum(anomaly_results['anomalies'])} anomalies ({np.mean(anomaly_results['anomalies'])*100:.1f}%)")
    
    # 2. Train PyTorch LSTM
    print("\n⚡ Training PyTorch LSTM model...")
    X_train = sequences[:800]
    y_train = targets[:800, 0]  # Predict first feature (pH)
    
    lstm_model, lstm_losses = train_pytorch_lstm_model(
        X_train, y_train, 
        input_size=data.shape[1], 
        hidden_size=32, 
        num_epochs=50
    )
    print(f"✅ LSTM training completed. Final loss: {lstm_losses[-1]:.6f}")
    
    # 3. Train GAN for data synthesis
    print("\n🎭 Training GAN for synthetic data generation...")
    gan = EnvironmentalGAN(input_dim=data.shape[1], latent_dim=50)
    gan_history = gan.train(data.values, epochs=1000, batch_size=32, sample_interval=500)
    
    # Generate synthetic data
    synthetic_data = gan.generate_synthetic_data(100)
    print(f"🧪 Generated {len(synthetic_data)} synthetic samples")
    
    # 4. Initialize Deep Q-Learning Agent
    print("\n🤖 Initializing Deep Q-Learning Agent...")
    dql_agent = DeepQLearningAgent(state_size=data.shape[1], action_size=4)
    print("✅ DQL Agent initialized for environmental control optimization")
    
    # 5. Create Transformer model
    print("\n🎪 Creating Transformer model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_model = TransformerPredictor(
        input_dim=data.shape[1], 
        d_model=64, 
        nhead=4, 
        num_layers=3
    ).to(device)
    print("✅ Transformer model initialized")
    
    # 6. Create CNN Classifier
    print("\n🔍 Creating CNN Time Series Classifier...")
    cnn_classifier = CNNTimeSeriesClassifier(
        input_channels=data.shape[1],
        sequence_length=24,
        num_classes=3  # Normal, Warning, Critical
    ).to(device)
    print("✅ CNN Classifier initialized")
    
    print("\n🎉 Comprehensive Neural Network Suite Created Successfully!")
    print(f"📋 Suite includes:")
    print(f"   - Autoencoder for anomaly detection")
    print(f"   - LSTM for time series prediction")
    print(f"   - GAN for synthetic data generation")
    print(f"   - Deep Q-Learning agent for optimization")
    print(f"   - Transformer for advanced forecasting")
    print(f"   - CNN for time series classification")
    
    return {
        'autoencoder': autoencoder,
        'lstm_model': lstm_model,
        'gan': gan,
        'dql_agent': dql_agent,
        'transformer': transformer_model,
        'cnn_classifier': cnn_classifier,
        'data': data,
        'scaler': scaler,
        'synthetic_data': synthetic_data,
        'anomaly_results': anomaly_results
    }

if __name__ == "__main__":
    # Create and demonstrate the neural network suite
    suite = create_comprehensive_neural_network_suite()
