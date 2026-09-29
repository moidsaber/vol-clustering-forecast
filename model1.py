import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

close = pd.read_csv("spx_close.csv", index_col=0, parse_dates=True)["Close"]

r = np.log(close / close.shift(1)).dropna()
vol = (100 * r.abs()).to_numpy()
window = 20
dates = r.index[window:]

x = np.stack([vol[i - window:i] for i in range(window,len(vol))])
y = vol[window:]

train = dates < "2021-01-01"
val   = (dates >= "2021-01-01") & (dates < "2023-01-01")
test  = dates >= "2023-01-01"

def to_tensors(mask):
    x_t = torch.tensor(x[mask], dtype=torch.float32)
    y_t = torch.tensor(y[mask], dtype=torch.float32).unsqueeze(1)
    return x_t, y_t

x_train, y_train = to_tensors(train)
x_val, y_val     = to_tensors(val)
x_test, y_text   = to_tensors(test)

x_v, y_v = x[val], y[val]
mse_today = np.mean((x_v[:, -1] - y_v) ** 2)
mse_avg = np.mean((x_v.mean(axis=1) - y_v) ** 2)

model = nn.Sequential(
    nn.Linear(window, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val, best_epoch, best_state = float("inf"), 0, None

for epoch in range(2000):
    optimizer.zero_grad()
    loss = loss_fn(model(x_train), y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val).item()
    if val_loss < best_val:
        best_val, best_epoch = val_loss, epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}   train {loss.item():.4f}   val {val_loss:.4f}")

model.load_state_dict(best_state)

print(f"\nval MSE  tomorrow = today : {mse_today:.4f}")
print(f"val MSE  20-day average   : {mse_avg:.4f}")
print(f"val MSE  MLP              : {best_val:.4f}  (best epoch {best_epoch})")

x_t, y_t = x[test], y[test]
with torch.no_grad():
    pred_mlp = model(x_test).squeeze(1).numpy()
pred_avg =x_t.mean(axis=1)

test_today = np.mean((x_t[:, -1] - y_t) ** 2)
test_avg = np.mean((pred_avg - y_t) ** 2)
test_mlp = np.mean((pred_mlp - y_t) ** 2)

print(f"\ntest MSE tomorrow = today : {test_today:.4f}")
print(f"test MSE 20-day average   : {test_avg:.4f}")
print(f"test MSE MLP              : {test_mlp:.4f}")
print(f"MLP / 20-day average      : {test_mlp / test_avg:.3f}  (below 1 means the MLP wins)")

# ----- Plot: forecasts vs realised, full test period plus a zoom ---------
test_dates = dates[test]
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
for ax in axes:
    ax.plot(test_dates, y_t, color="lightgray", lw=0.8, label="realised |r|")
    ax.plot(test_dates, pred_avg, lw=1.2, label="20-day average")
    ax.plot(test_dates, pred_mlp, lw=1.2, label="MLP")
    ax.set_ylabel("|daily return| (%)")
axes[0].set_title("Test period: forecasts vs realised")
axes[0].legend()
axes[1].set_xlim(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30"))
axes[1].set_title("Zoom: the April 2025 tariff shock")
plt.tight_layout()
plt.savefig("forecast_vs_realised.png", dpi=150)
plt.show()
