import tensorflow as tf
import numpy as np

# Add GPU memory management at the start of the file
try:
    # Attempt to configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured")
    else:
        print("No GPU devices found, running on CPU")
except Exception as e:
    print(f"GPU initialization failed: {e}")
    print("Falling back to CPU")
    # Instead of disabling CPU, we just disable GPU
    tf.config.set_visible_devices([], 'GPU')

class EnsembleNN:
    def __init__(self, n_models, input_shape, hidden_layers, output_shape):
        """
        Initialize ensemble of neural networks
        
        Args:
            n_models: Number of models in the ensemble
            input_shape: Shape of input data
            hidden_layers: List of integers representing neurons in hidden layers
            output_shape: Number of output neurons
        """
        self.n_models = n_models
        self.models = []
        self.optimizers = []  # Add list to store optimizers
        
        # Create n identical models
        for _ in range(n_models):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=input_shape))
            
            # Add hidden layers
            for units in hidden_layers:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
            
            # Add output layer
            model.add(tf.keras.layers.Dense(output_shape))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mse')
            
            # Build the model with sample data
            sample_data = tf.zeros((1,) + input_shape)
            model(sample_data)
            
            # Create an optimizer for this model
            self.optimizers.append(tf.keras.optimizers.Adam())
            self.models.append(model)

    def split_data(self, X, y):
        """Split data into n chunks for n models"""
        chunk_size = len(X) // self.n_models
        X_chunks = []
        y_chunks = []
        
        for i in range(self.n_models):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_models - 1 else len(X)
            X_chunks.append(X[start_idx:end_idx])
            y_chunks.append(y[start_idx:end_idx])
            
        return X_chunks, y_chunks

    def train_step(self, X_chunks, y_chunks, loss_fn):
        """Perform one training step with gradient averaging"""
        all_gradients = []

        # Convert inputs to tensors if they aren't already
        X_chunks = [tf.convert_to_tensor(X, dtype=tf.float32) for X in X_chunks]
        y_chunks = [tf.convert_to_tensor(y, dtype=tf.float32) for y in y_chunks]

        # Compute gradients for each model
        for model, X_chunk, y_chunk in zip(self.models, X_chunks, y_chunks):
            with tf.GradientTape() as tape:
                predictions = model(X_chunk, training=True)
                loss = loss_fn(y_chunk, predictions)
            
            # Get gradients for current model
            gradients = tape.gradient(loss, model.trainable_variables)
            all_gradients.append(gradients)

        # Average gradients across all models
        avg_gradients = [
            tf.reduce_mean([grad[i] for grad in all_gradients], axis=0)
            for i in range(len(all_gradients[0]))
        ]

        # Apply averaged gradients to all models using their respective optimizers
        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.apply_gradients(zip(avg_gradients, model.trainable_variables))

    def fit(self, X, y, epochs=10, batch_size=32):
        """Train the ensemble"""
        # Convert inputs to tensors if they aren't already
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        X_chunks, y_chunks = self.split_data(X, y)
        loss_fn = tf.keras.losses.MeanSquaredError()

        for epoch in range(epochs):
            self.train_step(X_chunks, y_chunks, loss_fn)
            
            # Calculate and print average loss across all models
            total_loss = 0
            for model, X_chunk, y_chunk in zip(self.models, X_chunks, y_chunks):
                predictions = model(X_chunk, training=False)
                total_loss += loss_fn(y_chunk, predictions)
            avg_loss = total_loss / self.n_models
            
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def predict(self, X):
        """Make predictions using the ensemble"""
        predictions = []
        for model in self.models:
            pred = model(X, training=False)
            predictions.append(pred)
        
        # Average predictions from all models
        return tf.reduce_mean(predictions, axis=0) 