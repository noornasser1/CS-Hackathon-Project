from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


from myData import anti_outputs
from myData import amplitudes
from myData import outputs, samples, num_frames, sample_rate, duration, time



original_waves = amplitudes
inverse_waves = anti_outputs

# Flatten each sample for use with SVR
original_waves_flat = original_waves.reshape(original_waves.shape[0], -1)
inverse_waves_flat = inverse_waves.reshape(inverse_waves.shape[0], -1)

# Normalize data
scaler = StandardScaler()
original_waves_flat = scaler.fit_transform(original_waves_flat)
inverse_waves_flat = scaler.transform(inverse_waves_flat)

# Split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(original_waves_flat, inverse_waves_flat, test_size=0.2)



# Train model
model = SVR()
model.fit(train_X, train_y)

# Evaluate model
train_preds = model.predict(train_X)
test_preds = model.predict(test_X)
print(train_preds)

train_loss = mean_squared_error(train_y, train_preds)
test_loss = mean_squared_error(test_y, test_preds)

print(f'Train loss: {train_loss}')
print(f'Test loss: {test_loss}')


# Choose start and end indices for the slice
start_index = 0
end_index = 1000  # Adjust this as needed

plt.figure(figsize=(10, 5))
plt.plot(test_preds[start_index:end_index], label='Predictions')
plt.plot(test_y[start_index:end_index], label='Ground Truth')
plt.legend()
plt.show()


