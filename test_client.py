import logging
import requests
import json
import base64
from io import BytesIO
from PIL import Image # 👈 Thêm thư viện này

# Configure simple logging for the client script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_client")

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
        
        logger.info(f"Đang gửi request đến {ML_SERVICE_URL}...")
        response = requests.post(ML_SERVICE_URL, files=files, data=data)
        
        # Kiểm tra response
        if response.status_code == 200:
            logger.info("--- Response thành công (200 OK) ---")
            response_data = response.json()
            
            logger.info(f"Temp ID: {response_data.get('temp_id')}")
            logger.info(f"Input Metadata: {response_data.get('input_metadata')}")
            logger.info("\n--- Dự đoán ---")
            logger.info(f"Chẩn đoán chính: {response_data['prediction']['class_name']}")
            logger.info(f"Độ tin cậy: {response_data['prediction']['score']:.4f}")
            
            logger.info("\n--- Tất cả điểm số ---")
            for score in response_data['all_scores'][:3]: # Hiển thị top 3
                logger.info(f"- {score['class_name']}: {score['score']:.4f}")
                
            logger.info("\n--- XAI (Grad-CAM) ---")
            
            # --- PHẦN MỚI ĐỂ MỞ ẢNH ---
            heatmap_base64 = response_data['xai_explanation']['heatmap_base64']
            if heatmap_base64:
                logger.info("Đã nhận được ảnh heatmap. Đang giải mã và mở...")
                
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
                    logger.exception(f"Lỗi khi mở ảnh heatmap: {e}")
            # --- KẾT THÚC PHẦN MỚI ---
            else:
                logger.info("Không nhận được heatmap.")

        else:
            logger.error(f"LỖI: Service trả về status code {response.status_code}")
            logger.error(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    logger.error(f"LỖI: Không thể kết nối đến {ML_SERVICE_URL}. Service ML đã chạy chưa?")
except FileNotFoundError:
    logger.error(f"LỖI: Không tìm thấy file ảnh test tại {TEST_IMAGE_PATH}. Hãy cập nhật đường dẫn.")
except Exception as e:
    logger.exception(f"Lỗi không xác định: {e}")