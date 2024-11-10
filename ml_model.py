import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# demo data
class HeartData(Dataset):
    def __init__(self, impedance, activity, target_delay):
        self.impedance = torch.tensor(impedance, dtype=torch.float32)
        self.activity = torch.tensor(activity, dtype=torch.float32)
        self.target_delay = torch.tensor(target_delay, dtype=torch.float32)

    def __len__(self):
        return len(self.impedance)

    def __getitem__(self, idx):
        return self.impedance[idx], self.activity[idx], self.target_delay[idx]


class AVSyncModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=50, output_size=1):
        super(AVSyncModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, impedance, activity):
        x = torch.cat((impedance, activity), dim=-1)
        lstm_out, _ = self.lstm(x)
        final_out = self.fc(lstm_out[:, -1, :])
        return final_out


impedance_data = np.random.rand(1000, 10)
activity_data = np.random.rand(1000, 1)
target_delay_data = np.random.rand(1000, 1)

dataset = HeartData(impedance_data, activity_data, target_delay_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



model = AVSyncModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 20
for epoch in range(epochs):
    for impedance, activity, target_delay in dataloader:
        optimizer.zero_grad()
        outputs = model(impedance, activity)
        loss = criterion(outputs, target_delay)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print("Training completed.")


def predict_av_delay(model, impedance, activity):
    model.eval()
    with torch.no_grad():
        impedance_tensor = torch.tensor(impedance, dtype=torch.float32).unsqueeze(0)
        activity_tensor = torch.tensor(activity, dtype=torch.float32).unsqueeze(0)
        predicted_delay = model(impedance_tensor, activity_tensor)
    return predicted_delay.item()


# Example real-time sensor data
real_time_impedance = np.random.rand(1, 10)  # 1 sample, 10 features
real_time_activity = np.random.rand(1, 1)

predicted_delay = predict_av_delay(model, real_time_impedance, real_time_activity)
print(f"Predicted AV Delay: {predicted_delay:.2f} ms")