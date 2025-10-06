import serial
import csv

ser = serial.Serial('COM3', 115200) 
ser.flushInput()

num_samples = 10 * 250 
collected_data = []

print("Bắt đầu thu thập dữ liệu...")
for _ in range(num_samples):
    try:
        line = ser.readline()
        # Chuyển đổi từ byte sang số nguyên
        data_point = int(line.decode('utf-8').strip())
        collected_data.append(data_point)
    except:
        pass # Bỏ qua nếu có lỗi đọc
print("Thu thập hoàn tất!")

# Lưu vào file CSV
with open('ecg_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for item in collected_data:
        writer.writerow([item])