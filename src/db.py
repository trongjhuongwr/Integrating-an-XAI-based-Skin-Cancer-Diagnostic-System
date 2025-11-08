import os
import uuid
import io
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
print(f"Supabase URL: {SUPABASE_URL}")

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print(f"Supabase Key: {SUPABASE_KEY}")

# Optional: a service role key with elevated privileges (DO NOT expose to client/browser)
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
if SUPABASE_SERVICE_ROLE:
    print("Supabase service role key found in env.")
else:
    print("No Supabase service role key found in env. Writes may fail due to RLS.")

# Primary client (typically anon/public key)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Privileged client for server-side writes (if provided)
service_supabase: Client | None = None
if SUPABASE_SERVICE_ROLE:
    try:
        service_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
    except Exception as e:
        print(f"Failed to create service_supabase client: {e}")
        service_supabase = None


def _get_write_client():
    """Return a supabase client suitable for write operations.

    Prefer the service role client (has elevated privileges). If not available,
    return the primary client but warn that RLS may block inserts/uploads.
    """
    if service_supabase is not None:
        return service_supabase
    return supabase


def quick_save(patient_info, img_pil, gradcam_pil, lime_pil, prediction):
    """
    Lưu toàn bộ kết quả (ảnh, dự đoán, XAI) vào Supabase.
    Nếu patient_code đã tồn tại → tái sử dụng patient_id.
    """
    # Kiểm tra patient_code 
    code = patient_info.get("code")
    # Use read via primary client (ok for most DB reads) but fallback to service client if needed
    read_client = supabase if supabase is not None else service_supabase
    patient_query = read_client.table("patients").select("*").eq("patient_code", code).execute()

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
    write_client = _get_write_client()
    try:
        write_client.storage.from_("isic2019-images").upload(path=image_path, file=img_bytes.getvalue(), file_options={"content-type": "image/jpeg"})
        public_url = write_client.storage.from_("isic2019-images").get_public_url(image_path)
    except Exception as e:
        # Surface a helpful error that explains RLS/service role issues
        msg = getattr(e, 'args', [str(e)])[0]
        raise RuntimeError(
            f"Failed to upload image to Supabase storage: {msg}.\n"
            "This commonly happens when Row Level Security (RLS) or storage policies prevent writes for the anon key.\n"
            "Fixes: (1) provide SUPABASE_SERVICE_ROLE env var to the server so writes use the service role key,\n"
            "or (2) adjust your Supabase table/storage policies to allow inserts from the anon key.\n"
            "NOTE: the service role key is highly privileged and must NOT be exposed in browser/client code."
        ) from e

    # Lesion 
    # Use privileged client for inserts to avoid RLS restrictions when possible
    write_client = _get_write_client()
    lesion_res = write_client.table("lesions").insert({
        "lesion_code": f"LS-{uuid.uuid4().hex[:6]}",
        "anatom_site_general": "unknown",
        "patient_id": patient_id
    }).execute()
    lesion_id = lesion_res.data[0]["id"]

    # Image 
    img_record = write_client.table("patient_img").insert({
        "patient_id": patient_id,
        "lesion_id": lesion_id,
        "image_url": public_url,
        "image_name": image_name
    }).execute()
    image_id = img_record.data[0]["id"]

    # Model Prediction
    # Some DB schemas require created_at (non-null). Provide server-side timestamp
    now_iso = datetime.utcnow().isoformat() + "Z"
    pred_record = write_client.table("model_predictions").insert({
        "image_id": image_id,
        "predicted_label": prediction["label"],
        "confidence": float(prediction["confidence"]),
        "model_version": prediction.get("model_version", "v1.0"),
        "created_at": now_iso
    }).execute()
    predict_id = pred_record.data[0]["id"]

    # Upload GradCAM & LIME
    for xai_img, xai_type in [(gradcam_pil, "Grad-CAM"), (lime_pil, "LIME")]:
        xai_bytes = io.BytesIO()
        xai_img.save(xai_bytes, format="JPEG")
        xai_bytes.seek(0)

        xai_path = f"{xai_type}/{uuid.uuid4()}.jpg"
        try:
            write_client.storage.from_("XAI_results").upload(path=xai_path, file=xai_bytes.getvalue(), file_options={"content-type": "image/jpeg"})
            xai_public_url = write_client.storage.from_("XAI_results").get_public_url(xai_path)
        except Exception as e:
            msg = getattr(e, 'args', [str(e)])[0]
            raise RuntimeError(
                f"Failed to upload XAI image to Supabase storage: {msg}.\n"
                "Check storage policies or provide SUPABASE_SERVICE_ROLE for server-side writes."
            ) from e

        write_client.table("xai_explainations").insert({
            "xai_type": xai_type,
            "explaination_json": {"type": xai_type},
            "explaination_image_url": xai_public_url,
            "predict_id": predict_id
        }).execute()

    # Diagnosis
    diag_record = write_client.table("diagnosis_results").insert({
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
    try:
        query = supabase.rpc("get_patient_records").execute()
        return query.data
    except Exception:
        # If anon client cannot run the RPC due to RLS, try the service client
        if service_supabase is not None:
            query = service_supabase.rpc("get_patient_records").execute()
            return query.data
        # Otherwise re-raise with helpful message
        raise RuntimeError(
            "Failed to call RPC get_patient_records. This may be due to RLS policies blocking anon access.\n"
            "If you expect this RPC to be public, adjust the function policies in Supabase, or provide SUPABASE_SERVICE_ROLE for server-side reads."
        )
