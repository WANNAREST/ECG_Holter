import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose
import matplotlib.pyplot as plt
import wfdb # Thư viện để đọc dữ liệu từ PhysioNet

# Tải và Chuẩn bị Dữ liệu ---
def load_and_prepare_data(record_name='118', noise_type='ma'):
    """Tải dữ liệu sạch và nhiễu từ MIT-BIH Noise Stress Test Database."""
    # Tải bản ghi, bao gồm tín hiệu gốc và nhiễu
    record = wfdb.rdrecord(f'nstdb/{record_name}', sampfrom=0, sampto=30000)
    noise_record = wfdb.rdrecord(f'nstdb/{noise_type}', sampfrom=0, sampto=30000)

    # Lấy tín hiệu ECG sạch từ kênh 1
    clean_signal = record.p_signal[:, 0]
    # Lấy tín hiệu nhiễu và thêm vào tín hiệu sạch
    noise = noise_record.p_signal[:, 0]
    noisy_signal = clean_signal + noise

    # Chuẩn hóa dữ liệu
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)

    clean_signal = normalize(clean_signal)
    noisy_signal = normalize(noisy_signal)
    
    return noisy_signal, clean_signal

def create_segments(noisy, clean, segment_len=512):
    """Cắt tín hiệu thành các đoạn dài 512 mẫu."""
    num_segments = len(noisy) // segment_len
    noisy_segments = []
    clean_segments = []
    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len
        noisy_segments.append(noisy[start:end])
        clean_segments.append(clean[start:end])
    
    # Reshape cho đầu vào của CNN: (số lượng, độ dài, số kênh)
    return np.array(noisy_segments)[..., np.newaxis], np.array(clean_segments)[..., np.newaxis]

# Tải và chuẩn bị dữ liệu
noisy_signal, clean_signal = load_and_prepare_data()
X_train, y_train = create_segments(noisy_signal, clean_signal)

print("Kích thước dữ liệu huấn luyện (Input):", X_train.shape)
print("Kích thước dữ liệu huấn luyện (Output):", y_train.shape)


#  Xây dựng Mô hình 1D-CNN Autoencoder ---
def build_autoencoder_model(input_shape=(512, 1)):
    """Xây dựng kiến trúc mô hình dựa trên Hình 2 của bài báo."""
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    # Bottleneck
    x = Conv1D(256, 5, activation='relu', padding='same')(x)

    # Decoder
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(128, 5, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(64, 5, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(32, 5, activation='relu', padding='same')(x)
    
    # Output Layer
    outputs = Conv1DTranspose(1, 5, activation='linear', padding='same')(x)
    
    model = Model(inputs, outputs)
    return model

# Khởi tạo mô hình
model = build_autoencoder_model()
model.summary()

#  Huấn luyện Mô hình ---
model.compile(optimizer='adam', loss='mean_squared_error')

# Thêm EarlyStopping để tránh overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nBắt đầu huấn luyện mô hình...")
history = model.fit(X_train, y_train,
                    epochs=100, # Giảm số epochs để chạy demo nhanh hơn
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# Đánh giá và Trực quan hóa Kết quả ---
print("\nĐánh giá trên một mẫu thử...")
# Lấy một đoạn tín hiệu để kiểm tra
test_noisy = X_train[10:11]
test_clean = y_train[10:11]

# Dự đoán tín hiệu sạch
denoised_signal = model.predict(test_noisy)

# Tính toán SNR
def calculate_snr(clean, denoised):
    noise_power = np.sum((clean - denoised) ** 2)
    signal_power = np.sum(clean ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_val = calculate_snr(test_clean.flatten(), denoised_signal.flatten())
print(f"Signal-to-Noise Ratio (SNR) trên mẫu thử: {snr_val:.2f} dB")


# Vẽ kết quả
plt.figure(figsize=(15, 6))
plt.plot(test_noisy.flatten(), 'b', label='Tín hiệu Nhiễu (Input)', alpha=0.6)
plt.plot(test_clean.flatten(), 'g', label='Tín hiệu Sạch (Ground Truth)', linewidth=2)
plt.plot(denoised_signal.flatten(), 'r', label='Tín hiệu Đã khử nhiễu (Model Output)', linewidth=2, linestyle='--')
plt.title('So sánh Tín hiệu ECG')
plt.xlabel('Mẫu (Sample)')
plt.ylabel('Biên độ Chuẩn hóa')
plt.legend()
plt.grid(True)
plt.show()