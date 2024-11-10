import smbus
import time
import torch
import numpy as np
import ml_model

# Load model on Raspberry Pi
model = ml_model.AVSyncModel()
model.load_state_dict(torch.load('av_sync_model.pt'))
model.eval()

# Setup MPU6050 accelerometer (assuming I2C address 0x68)
bus = smbus.SMBus(1)  # I2C bus
MPU6050_ADDR = 0x68


def read_accelerometer():
    # Read accelerometer data (example for MPU6050)
    accel_x = bus.read_word_data(MPU6050_ADDR, 0x3B)
    accel_y = bus.read_word_data(MPU6050_ADDR, 0x3D)
    accel_z = bus.read_word_data(MPU6050_ADDR, 0x3F)
    return np.array([accel_x, accel_y, accel_z]) / 16384.0  # Normalize


# Mock impedance sensor function (replace with actual reading code)
def read_impedance():
    # Placeholder for actual impedance reading
    return np.random.rand(10)  # Dummy data for demonstration


# Real-time prediction function
def predict_av_delay():
    impedance = read_impedance()
    activity = read_accelerometer()

    impedance_tensor = torch.tensor(impedance, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    activity_tensor = torch.tensor(activity, dtype=torch.float32).unsqueeze(0).unsqueeze(
        0)  # Add batch and sequence dimensions

    with torch.no_grad():
        predicted_delay = model(impedance_tensor, activity_tensor)
    return predicted_delay.item()


# Loop to continuously make predictions
while True:
    av_delay = predict_av_delay()
    print(f"Predicted AV Delay: {av_delay:.2f} ms")
    time.sleep(1)  # Delay for real-time responsiveness
