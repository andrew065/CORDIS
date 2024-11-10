import smbus
import time
import torch
import numpy as np
import ml_model


model = ml_model.AVSyncModel()
model.load_state_dict(torch.load('av_sync_model.pt'))
model.eval()

# MPU6050 accelerometer (assuming I2C address 0x68)
bus = smbus.SMBus(1)
MPU6050_ADDR = 0x68


def read_accelerometer():
    accel_x = bus.read_word_data(MPU6050_ADDR, 0x3B)
    accel_y = bus.read_word_data(MPU6050_ADDR, 0x3D)
    accel_z = bus.read_word_data(MPU6050_ADDR, 0x3F)
    return np.array([accel_x, accel_y, accel_z]) / 16384.0


# Demo impedance sensor function
def read_impedance():
    return np.random.rand(10)


def predict_av_delay():
    impedance = read_impedance()
    activity = read_accelerometer()

    impedance_tensor = torch.tensor(impedance, dtype=torch.float32).unsqueeze(0)
    activity_tensor = torch.tensor(activity, dtype=torch.float32).unsqueeze(0).unsqueeze(
        0)

    with torch.no_grad():
        predicted_delay = model(impedance_tensor, activity_tensor)
    return predicted_delay.item()


while True:
    av_delay = predict_av_delay()
    print(f"Predicted AV Delay: {av_delay:.2f} ms")
    time.sleep(1)  # Delay for real-time responsiveness
