import requests
import json
import base64
from io import BytesIO
from PIL import Image # 👈 Thêm thư viện này

# URL của service ML (thay đổi nếu cần)
ML_SERVICE_URL = "http://localhost:8000/api/v1/predict"

# Đường dẫn đến một ảnh test (thay bằng ảnh của bạn)
TEST_IMAGE_PATH = "C:/Users/ASUS/Downloads/ISIC_0034321.jpg"

# Dữ liệu metadata mẫu
metadata = {
    "temp_id": "test-12345",
    "age": 55,
    "gender": "male",
    "lesion_location": "back"
}

# Chuyển metadata thành chuỗi JSON
metadata_str = json.dumps(metadata)

try:
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {
            'image': (TEST_IMAGE_PATH.split('/')[-1], f, 'image/jpeg')
        }
        data = {
            'metadata': metadata_str
        }
        
        print(f"Đang gửi request đến {ML_SERVICE_URL}...")
        
        response = requests.post(ML_SERVICE_URL, files=files, data=data)
        
        # Kiểm tra response
        if response.status_code == 200:
            print("--- Response thành công (200 OK) ---")
            response_data = response.json()
            
            print(f"Temp ID: {response_data.get('temp_id')}")
            print(f"Input Metadata: {response_data.get('input_metadata')}")
            print("\n--- Dự đoán ---")
            print(f"Chẩn đoán chính: {response_data['prediction']['class_name']}")
            print(f"Độ tin cậy: {response_data['prediction']['score']:.4f}")
            
            print("\n--- Tất cả điểm số ---")
            for score in response_data['all_scores'][:3]: # Hiển thị top 3
                print(f"- {score['class_name']}: {score['score']:.4f}")
                
            print("\n--- XAI (Grad-CAM) ---")
            
            # --- PHẦN MỚI ĐỂ MỞ ẢNH ---
            heatmap_base64 = response_data['xai_explanation']['heatmap_base64']
            if heatmap_base64:
                print("Đã nhận được ảnh heatmap. Đang giải mã và mở...")
                
                # Chuỗi base64 có dạng "data:image/png;base64,iVBOR..."
                # Chúng ta cần tách phần data ra
                try:
                    header, encoded = heatmap_base64.split(",", 1)
                    decoded_data = base64.b64decode(encoded)
                    
                    # Tạo ảnh từ dữ liệu bytes
                    image = Image.open(BytesIO(decoded_data))
                    
                    # Mở ảnh bằng trình xem ảnh mặc định
                    image.show()
                    
                except Exception as e:
                    print(f"Lỗi khi mở ảnh heatmap: {e}")
            # --- KẾT THÚC PHẦN MỚI ---
            else:
                print("Không nhận được heatmap.")

        else:
            print(f"LỖI: Service trả về status code {response.status_code}")
            print(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    print(f"LỖI: Không thể kết nối đến {ML_SERVICE_URL}. Service ML đã chạy chưa?")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file ảnh test tại {TEST_IMAGE_PATH}. Hãy cập nhật đường dẫn.")
except Exception as e:
    print(f"Lỗi không xác định: {e}")