import numpy as np
from ensemble_nn import EnsembleNN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Generate sample data
X = np.random.random((10000, 10))  # 1000 samples, 10 features
y = np.random.random((10000, 1))   # 1000 samples, 1 output

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ensemble with 3 networks
ensemble = EnsembleNN(
    n_models=3,
    input_shape=(10,),
    hidden_layers=[32],
    output_shape=1
)

# Create single network with same architecture
single_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
single_model.compile(optimizer='adam', loss='mse')

# Train both models
print("\nTraining Ensemble:")
ensemble.fit(X_train, y_train, epochs=10, batch_size=32)

print("\nTraining Single Network:")
single_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

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