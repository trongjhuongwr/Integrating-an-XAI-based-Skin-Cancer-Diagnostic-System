import os
import uuid
import io
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
print(f"Supabase URL: {SUPABASE_URL}")

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print(f"Supabase Key: {SUPABASE_KEY}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def quick_save(patient_info, img_pil, gradcam_pil, lime_pil, prediction):
    """
    Lưu toàn bộ kết quả (ảnh, dự đoán, XAI) vào Supabase.
    Nếu patient_code đã tồn tại → tái sử dụng patient_id.
    """
    # Kiểm tra patient_code 
    code = patient_info.get("code")
    patient_query = supabase.table("patients").select("*").eq("patient_code", code).execute()

    if patient_query.data:
        patient_id = patient_query.data[0]["id"]
    else:
        # Tạo mới nếu chưa có
        patient_res = supabase.table("patients").insert({
            "patient_code": code,
            "sex": patient_info.get("sex", "Unknown"),
            "age_approx": patient_info.get("age", 0),
        }).execute()
        patient_id = patient_res.data[0]["id"]

    # Upload ảnh gốc 
    class_folder = prediction["label"]
    image_name = f"{uuid.uuid4()}.jpg"
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    image_path = f"{class_folder}/{image_name}"
    supabase.storage.from_("isic2019-images").upload(path=image_path, file=img_bytes.getvalue(), file_options={"content-type": "image/jpeg"})
    public_url = supabase.storage.from_("isic2019-images").get_public_url(image_path)

    # Lesion 
    lesion_res = supabase.table("lesions").insert({
        "lesion_code": f"LS-{uuid.uuid4().hex[:6]}",
        "anatom_site_general": "unknown",
        "patient_id": patient_id
    }).execute()
    lesion_id = lesion_res.data[0]["id"]

    # Image 
    img_record = supabase.table("patient_img").insert({
        "patient_id": patient_id,
        "lesion_id": lesion_id,
        "image_url": public_url,
        "image_name": image_name
    }).execute()
    image_id = img_record.data[0]["id"]

    # Model Prediction
    pred_record = supabase.table("model_predictions").insert({
        "image_id": image_id,
        "predicted_label": prediction["label"],
        "confidence": float(prediction["confidence"]),
        "model_version": prediction.get("model_version", "v1.0")
    }).execute()
    predict_id = pred_record.data[0]["id"]

    # Upload GradCAM & LIME
    for xai_img, xai_type in [(gradcam_pil, "Grad-CAM"), (lime_pil, "LIME")]:
        xai_bytes = io.BytesIO()
        xai_img.save(xai_bytes, format="JPEG")
        xai_bytes.seek(0)

        xai_path = f"{xai_type}/{uuid.uuid4()}.jpg"
        supabase.storage.from_("XAI_results").upload(path=xai_path, file=xai_bytes.getvalue(), file_options={"content-type": "image/jpeg"})
        xai_public_url = supabase.storage.from_("XAI_results").get_public_url(xai_path)

        supabase.table("xai_explainations").insert({
            "xai_type": xai_type,
            "explaination_json": {"type": xai_type},
            "explaination_image_url": xai_public_url,
            "predict_id": predict_id
        }).execute()

    # Diagnosis
    diag_record = supabase.table("diagnosis_results").insert({
        "image_id": image_id,
        "diagnosis_type": prediction["label"]
    }).execute()
    diagnosis_id = diag_record.data[0]["id"]

    return {
        "patient_id": patient_id,
        "image_id": image_id,
        "predict_id": predict_id,
        "diagnosis_id": diagnosis_id,
        "image_url": public_url
    }

def get_patient_records():
    """
    Lấy danh sách bệnh nhân đã có kèm kết quả chẩn đoán mới nhất.
    """
    query = supabase.rpc("get_patient_records").execute()
    return query.data
