import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, find_peaks
from ecg_preprocessing import fs,final_signal

# LỌC DẢI THÔNG (BANDPASS FILTER) ---
# Tối ưu cho việc phát hiện QRS (5-15 Hz)
lowcut_pt = 5.0
highcut_pt = 15.0
nyquist = 0.5 * fs
low = lowcut_pt / nyquist
high = highcut_pt / nyquist
b_bandpass, a_bandpass = butter(1, [low, high], btype='band')
signal_bandpassed = filtfilt(b_bandpass, a_bandpass, final_signal)

#  PHÉP LẤY ĐẠO HÀM (DIFFERENTIATION) 
# Tìm tốc độ biến thiên (độ dốc)
signal_differentiated = np.diff(signal_bandpassed)

#  PHÉP BÌNH PHƯƠNG (SQUARING) 
# Khuếch đại các đỉnh QRS
signal_squared = signal_differentiated ** 2

#  TÍCH PHÂN CỬA SỔ TRƯỢT (MOVING-WINDOW INTEGRATION)
# Tổng hợp năng lượng để tạo ra đỉnh sóng rõ ràng
window_width = int(0.150 * fs) # Cửa sổ 150ms
signal_integrated = np.convolve(signal_squared, np.ones(window_width)/window_width, mode='same')

#  RA QUYẾT ĐỊNH (PHÁT HIỆN ĐỈNH R)
# Sử dụng thư viện scipy.signal.find_peaks để phát hiện đỉnh
# Đây là một cách đơn giản hóa bước ngưỡng động của Pan-Tompkins
# nhưng rất hiệu quả trong thực tế.
# Ngưỡng phát hiện đỉnh có thể cần được điều chỉnh tùy theo tín hiệu
# Ở đây, ta đặt ngưỡng bằng 50% của giá trị trung bình của các đỉnh tiềm năng
peak_threshold = 0.5 * np.mean(signal_integrated)

# Khoảng cách tối thiểu giữa các đỉnh (để tránh phát hiện nhầm T-wave)
# Giả sử nhịp tim tối đa là 200 bpm -> khoảng cách tối thiểu là 0.3s
min_peak_distance = int(0.3 * fs)

r_peaks, _ = find_peaks(signal_integrated, height=peak_threshold, distance=min_peak_distance)

plt.figure(figsize=(15, 7))

# Vẽ các bước xử lý của Pan-Tompkins
plt.subplot(2, 1, 1)
plt.plot(signal_integrated, label='Tín hiệu sau Tích phân')
plt.plot(r_peaks, signal_integrated[r_peaks], 'x', color='red', label='Đỉnh R tiềm năng')
plt.title('Tín hiệu đã xử lý bởi Pan-Tompkins')
plt.legend()
plt.grid(True)

# Vẽ tín hiệu ECG gốc và các đỉnh R đã phát hiện
plt.subplot(2, 1, 2)
plt.plot(final_signal, label='Tín hiệu ECG đã làm sạch')
plt.plot(r_peaks, final_signal[r_peaks], 'x', color='red', markersize=10, label='Đỉnh R đã phát hiện')
plt.title('Phát hiện đỉnh R trên Tín hiệu ECG')
plt.xlabel('Mẫu (Sample)')
plt.ylabel('Biên độ ADC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Tính toán khoảng RR (tính bằng giây)
rr_intervals_ms = np.diff(r_peaks) * (1000.0 / fs)

# Tính nhịp tim trung bình
if len(rr_intervals_ms) > 0:
    mean_rr_ms = np.mean(rr_intervals_ms)
    heart_rate_bpm = 60000.0 / mean_rr_ms
    print(f"Đã phát hiện {len(r_peaks)} đỉnh R.")
    print(f"Nhịp tim trung bình: {heart_rate_bpm:.2f} BPM")
else:
    print("Không phát hiện được đỉnh R nào.")