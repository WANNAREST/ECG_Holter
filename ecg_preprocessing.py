import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt

# --- Thông số ---
fs = 250.0  # Tần số lấy mẫu (Hz), phải khớp với delay trong code ESP32
f0 = 50.0   # Tần số nhiễu đường dây (Hz)
Q = 30.0    # Quality factor cho bộ lọc notch

raw_signal = np.loadtxt('ecg_data.csv')

#  Lọc nhiễu đường dây 50Hz (Notch Filter) 
b_notch, a_notch = iirnotch(f0, Q, fs)
signal_notched = filtfilt(b_notch, a_notch, raw_signal)

# Lọc trôi đường nền (High-pass Filter) 
lowcut = 0.5 # Tần số cắt (Hz)
order = 2
nyquist = 0.5 * fs
low = lowcut / nyquist
b_hp, a_hp = butter(order, low, btype='highpass')
signal_baseline_removed = filtfilt(b_hp, a_hp, signal_notched)

# Lọc nhiễu tần số cao (Low-pass Filter) ---
highcut = 40.0 # Tần số cắt (Hz)
high = highcut / nyquist
b_lp, a_lp = butter(order, high, btype='lowpass')
final_signal = filtfilt(b_lp, a_lp, signal_baseline_removed)

# --- Trực quan hóa kết quả ---
plt.figure(figsize=(12, 6))
plt.plot(raw_signal, label='Tín hiệu Gốc (Raw)')
plt.plot(final_signal, label='Tín hiệu Đã xử lý', color='red', linewidth=1.5)
plt.title('So sánh Tín hiệu ECG Trước và Sau khi Xử lý')
plt.legend()
plt.grid(True)
plt.show()