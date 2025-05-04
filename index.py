import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Bản đồ nhãn số thành tên loại rác
label_map = {
    1: 'cardboard',
    2: 'glass',
    3: 'metal',
    4: 'paper',
    5: 'plastic'
}

# Load mô hình .h5
model = load_model('trash.h5')

# Hàm tiền xử lý ảnh
def preprocess_image(image_path, target_size=(180, 180)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # chuẩn hóa nếu mô hình yêu cầu
    image_array = np.expand_dims(image_array, axis=0)  # thêm batch dimension
    return image_array

# Dự đoán và hiển thị ảnh
def predict_image(image_path):
    # Tiền xử lý ảnh
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    
    # Nếu là mô hình softmax cuối cùng:
    predicted_index = np.argmax(predictions[0])
    predicted_label_number = predicted_index + 1  # vì bạn đánh số từ 1-5
    predicted_label_name = label_map[predicted_label_number]
    
    # Đọc lại ảnh gốc để hiển thị
    original_image = Image.open(image_path).convert('RGB')
    original_image = np.array(original_image)
    
    # Hiển thị ảnh với nhãn dự đoán
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title(f'Prediction: {predicted_label_name} (Label: {predicted_label_number})', fontsize=12, color='blue')
    plt.axis('off')  # Tắt trục tọa độ
    plt.show()  # Hiển thị cửa sổ ảnh trực tiếp
    
    print(f'Dự đoán: {predicted_label_name} (Nhãn số: {predicted_label_number})')

# Ví dụ sử dụng
predict_image('paper.png')