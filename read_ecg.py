import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt


# Tải lại toàn bộ cấu trúc và trọng số của mô hình
trained_model = tf.keras.models.load_model('ecg_denoising_model.h5')
print("Tải mô hình thành công.")


# --- 2. Chuẩn bị dữ liệu mới từ ecg_data.csv ---
def prepare_new_data(filepath='ecg_data.csv', fs=250.0, segment_len=512):
  
    raw_signal = np.loadtxt(filepath)
    
    f0 = 50.0
    Q = 30.0
    b_notch, a_notch = iirnotch(f0, Q, fs)
    notched_signal = filtfilt(b_notch, a_notch, raw_signal)
    
    # Chuẩn hóa
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)
    
    input_signal = normalize(notched_signal)
    
    # Cắt thành các đoạn
    num_segments = len(input_signal) // segment_len
    segments = [input_signal[i*segment_len:(i+1)*segment_len] for i in range(num_segments)]
    
    # Reshape cho đầu vào của CNN
    return np.array(segments)[..., np.newaxis], raw_signal

# Chuẩn bị dữ liệu từ file csv của bạn
input_segments, original_raw_signal = prepare_new_data()


print("Bắt đầu khử nhiễu cho dữ liệu mới...")
denoised_segments = trained_model.predict(input_segments)
print("Khử nhiễu hoàn tất.")

denoised_signal_full = denoised_segments.flatten()

plt.figure(figsize=(15, 6))

plot_range = range(0, 2000) 
plt.plot(original_raw_signal[plot_range], 'b', label='Tín hiệu Gốc từ ecg_data.csv', alpha=0.6)
plt.plot(denoised_signal_full[plot_range], 'r', label='Tín hiệu đã được AI khử nhiễu', linewidth=2)
plt.title('Kết quả Khử nhiễu bằng Mô hình AI đã Huấn luyện')
plt.xlabel('Mẫu (Sample)')
plt.ylabel('Biên độ')
plt.legend()
plt.grid(True)
plt.show()
