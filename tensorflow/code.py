import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load and preprocess the dataset
data = pd.read_csv("housing.csv")

# Separate the target column
target = data.pop('median_house_value')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# One-hot encode the 'ocean_proximity' column
encoder = OneHotEncoder(sparse=False)
proximity_encoded = encoder.fit_transform(data['ocean_proximity'].values.reshape(-1, 1))
data_encoded = data.drop(columns=['ocean_proximity'])
data_encoded = pd.concat([data_encoded, pd.DataFrame(proximity_encoded, columns=encoder.get_feature_names_out(['OP']))], axis=1)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(data_encoded)
y = target.values.reshape(-1, 1)

# Train the model
num_epochs = 200
batch_size = 32
learning_rate = 0.001
num_features = X.shape[1]
num_samples = X.shape[0]
num_batches = num_samples // batch_size

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

for epoch in range(num_epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Compute the training loss for the epoch
    y_pred_train = model(X, training=False)
    loss_train = loss_fn(y, y_pred_train)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_train.numpy():.4f}")

# Save the model
model.save("housing_model_tf")
