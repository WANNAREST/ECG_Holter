#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// --- Khai báo cho cảm biến MPU 6050 ---
Adafruit_MPU6050 mpu;

// --- Khai báo cho cảm biến ECG AD8232 ---
#define ECG_PIN 34   // Chân OUTPUT từ AD8232 (phải là chân có ADC)
#define LO_PLUS 27   // Chân Lead-off positive
#define LO_MINUS 14  // Chân Lead-off negative

// --- Biến cho việc quản lý thời gian (non-blocking) ---
unsigned long previousMpuMillis = 0;
const long mpuInterval = 500; // Đọc MPU 6050 mỗi 500ms

void setup() {
  Serial.begin(115200);
  
  // --- Khởi tạo cho ECG AD8232 ---
  pinMode(ECG_PIN, INPUT);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  Serial.println("Khởi tạo cảm biến ECG AD8232 xong!");

  // --- Khởi tạo cho MPU 6050 ---
  if (!mpu.begin()) {
    Serial.println("Không tìm thấy cảm biến MPU6050! Kiểm tra lại kết nối.");
    while (1) {
      delay(10);
    }
  }
  Serial.println("Đã tìm thấy MPU6050!");

  // Cài đặt các thông số cho MPU6050
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
  // --- Tác vụ 1: Đọc cảm biến ECG liên tục (tốc độ cao) ---
  readEcgSensor();

  // --- Tác vụ 2: Đọc cảm biến MPU 6050 định kỳ (tốc độ thấp) ---
  unsigned long currentMillis = millis();
  if (currentMillis - previousMpuMillis >= mpuInterval) {
    // Đã đến lúc đọc MPU, lưu lại thời gian này
    previousMpuMillis = currentMillis;
    readMpuSensor();
  }
}

// Hàm riêng để đọc ECG cho gọn
void readEcgSensor() {
  int loPlus = digitalRead(LO_PLUS);
  int loMinus = digitalRead(LO_MINUS);

  if (loPlus == 1 || loMinus == 1) {
    // Để tránh spam Serial, bạn có thể chỉ in khi trạng thái thay đổi
    // Ở đây ta tạm in ra giá trị 0 để dễ vẽ đồ thị
    Serial.println(0); 
  } else {
    // Đọc và in giá trị ECG
    int ecgValue = analogRead(ECG_PIN);  
    Serial.println(ecgValue);
  }
  // Delay nhỏ để ổn định tốc độ lấy mẫu ADC, tương đương khoảng 250Hz
  delay(4); 
}

// Hàm riêng để đọc MPU 6050 cho gọn
void readMpuSensor() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // In các giá trị MPU (thêm tiền tố để phân biệt với dữ liệu ECG)
  Serial.print("MPU -> Accel X: "); Serial.print(a.acceleration.x);
  Serial.print(", Y: "); Serial.print(a.acceleration.y);
  Serial.print(", Z: "); Serial.println(a.acceleration.z);

  Serial.print("MPU -> Gyro X: "); Serial.print(g.gyro.x);
  Serial.print(", Y: "); Serial.print(g.gyro.y);
  Serial.print(", Z: "); Serial.println(g.gyro.z);
  
  Serial.println("---------------------------------");
}
