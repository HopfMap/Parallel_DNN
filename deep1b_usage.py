import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ensemble_nn import EnsembleNN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the deep1b dataset
dataset = tfds.load('deep1b', split='database', shuffle_files=False)

# Print the first example to see its structure
first_example = next(iter(dataset))
print("Dataset features:", first_example.keys())
print("Example structure:", first_example)

# Convert dataset to numpy arrays and take 10% of the data
data_sample = []
count = 0
max_samples = int(1e6)  # Adjust this number based on your memory constraints
target_samples = int(max_samples * 0.1)  # 10% of the data

for example in dataset:
    if count >= target_samples:
        break
    # Let's print the first example's keys to debug
    if count == 0:
        print("First example keys:", example.keys())
        print("First example:", example)
    data_sample.append({
        'features': example['embedding'].numpy(),  # Using 'embedding' key
        'index': count  # Using count as index since we don't have an ID field
    })
    count += 1
    if count % 10000 == 0:
        print(f"Loaded {count} samples...")

# Convert to numpy arrays
X = np.stack([item['features'] for item in data_sample])
y = np.stack([item['index'] for item in data_sample])
y = y.reshape(-1, 1)  # Reshape to 2D array

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ensemble with 3 networks
input_dim = X_train.shape[1]  # Get input dimension from data
ensemble = EnsembleNN(
    n_models=8,
    input_shape=(input_dim,),
    hidden_layers=[512, 256, 64],  # Larger architecture for this complex dataset
    output_shape=1
)

# Create single network with same architecture
single_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
single_model.compile(optimizer='adam', loss='mse')

# Training parameters
n_epochs = 200

# Train both models
print("\nTraining Ensemble:")
ensemble.fit(X_train, y_train, epochs=n_epochs, batch_size=32)

print("\nTraining Single Network:")
single_model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, verbose=1)

# Make predictions with both models
ensemble_pred = ensemble.predict(X_test).numpy()
single_pred = single_model.predict(X_test)

# Calculate metrics for ensemble
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

# Calculate metrics for single network
single_r2 = r2_score(y_test, single_pred)
single_mse = mean_squared_error(y_test, single_pred)
single_mae = mean_absolute_error(y_test, single_pred)

print("\nEnsemble Test Set Metrics:")
print(f"R² Score: {ensemble_r2:.4f}")
print(f"Mean Squared Error: {ensemble_mse:.4f}")
print(f"Mean Absolute Error: {ensemble_mae:.4f}")

print("\nSingle Network Test Set Metrics:")
print(f"R² Score: {single_r2:.4f}")
print(f"Mean Squared Error: {single_mse:.4f}")
print(f"Mean Absolute Error: {single_mae:.4f}") 

print("\nMSE ensemble/MSE single =",format(ensemble_mse/single_mse,".2f"))