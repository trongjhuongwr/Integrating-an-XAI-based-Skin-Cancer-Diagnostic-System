import streamlit as st
import concurrent.futures
from PIL import Image
import cv2, numpy as np, pandas as pd, torch
import torchvision.transforms as transforms
from src.model import load_pretrained_resnet50_model, predict_prob
from src.xai import explain_with_GRAD_CAM, explain_with_LIME
from src.db import quick_save, get_patient_records
import random

# Setup 
MODEL = load_pretrained_resnet50_model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(DEVICE).eval()

CLASS_LABELS = {
    0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK',
    4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.set_page_config(layout="wide", page_title="AI Diagnosis & Explainability System")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def run_gradcam(model, img, transform, device, pred_class):
    target_layer = model.layer4[-1].conv3
    input_tensor = transform(img).unsqueeze(0).to(device)
    grayscale_cam = explain_with_GRAD_CAM(input_tensor.squeeze(0), model, target_layer, pred_class)

    # Resize Grad-CAM to original image size
    w, h = img.size  # PIL gives (width, height)
    grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def run_lime(model, img, transform, device):
    return explain_with_LIME(img, model, transform, device)

# --- Tabs ---
tab1, tab2 = st.tabs(["Diagnosis", "Records"])

st.markdown("""
    <style>
        .block-container {max-width: 1600px !important;}
        .predict-btn {
            background-color: #0066cc;
            color: white;
            font-weight: 600;
            font-size: 16px;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            width: 100%;
        }
        .predict-btn:hover {
            background-color: #004c99;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- TAB 1 ----------------------
with tab1:
    st.title("Explainable Diagnosis")
    st.markdown("### Case Context")
    uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"], key="file")
    
    col1, col2, col3, col4 = st.columns(4)
    name = col1.text_input("Patient Name")
    age = col2.number_input("Age", min_value=0, max_value=120)
    gender = col3.selectbox("Gender", ["Other", "Male", "Female"])
    patient_id = col4.text_input("Patient ID")
    if not uploaded_file:
        st.info("Please upload a lesion image above to continue the diagnostic process.")
        st.stop()

    # Zone 1: Case Context & Zone 2: Lesion Image
    left_col, right_col = st.columns([1.1, 2])

    with left_col:
        # st.markdown("### Zone 1: Case Context")
        # name = st.text_input("Patient Name")
        # age = st.number_input("Age", min_value=0, max_value=120)
        # gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        # patient_id = st.text_input("Patient ID")

        st.markdown("### Lesion Image")
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded lesion image", use_container_width=True)

        if st.button("Run AI Prediction", key="predict", use_container_width=True):
            probs = predict_prob(img, MODEL, transform, DEVICE)
            pred_class = int(probs.argmax())
            conf = float(probs.max())
            pred_label = CLASS_LABELS[pred_class]

            st.session_state.update({
                "probs": probs,
                "pred_label": pred_label,
                "conf": conf,
                "img": img,
                "gradcam_future": executor.submit(run_gradcam, MODEL, img, transform, DEVICE, pred_class),
                "lime_future": executor.submit(run_lime, MODEL, img, transform, DEVICE)
            })
            st.success(f"Prediction: {pred_label} ({conf:.2f})")

    with right_col:
        if "probs" in st.session_state:
            st.markdown("### Explainability & Sensitivity Analysis")

            probs = st.session_state["probs"]
            pred_label = st.session_state["pred_label"]
            conf = st.session_state["conf"]

            prob_df = pd.DataFrame({
                "Class": [CLASS_LABELS[i] for i in range(len(probs))],
                "Probability": [round(float(p), 4) for p in probs]
            }).sort_values("Probability", ascending=False)

            col_pred, col_exp = st.columns([1.1, 2.3])

            with col_pred:
                st.markdown(f"**Predicted Class:** `{pred_label}`")
                st.markdown(f"**Confidence:** `{conf:.2f}`")
                st.dataframe(prob_df, use_container_width=True, height=300)

            with col_exp:
                mode = st.radio("Explainability Mode", ["Grad-CAM", "LIME"], horizontal=True)
                future = st.session_state.get("gradcam_future" if mode == "Grad-CAM" else "lime_future")

                if future is None:
                    st.warning("Please run prediction first.")
                elif future.done():
                    st.image(future.result(), caption=f"{mode} Visualization", use_container_width=True)
                else:
                    st.info(f"Generating {mode}... please wait.")

    # Zone 4: Final Clinical Decision
    if "probs" in st.session_state:
        st.markdown("---")
        st.markdown("### Final Clinical Decision")

        col_decision1, col_decision2 = st.columns([2, 1])
        with col_decision1:
            final_diagnosis = st.text_area("Final Diagnosis (clinician’s conclusion)")
            notes = st.text_area("Clinical Notes / Observations")

        with col_decision2:
            decision_status = st.selectbox("Decision Status", ["Pending", "Confirmed", "Rejected"])
            if st.button("Save Decision", use_container_width=True):
                patient_info = {
                    "code": patient_id.strip() or f"PT{random.randint(10000,99999)}",
                    "sex": gender,
                    "age": age
                }

                gradcam_img = st.session_state["gradcam_future"].result() if st.session_state["gradcam_future"].done() else None
                lime_img = st.session_state["lime_future"].result() if st.session_state["lime_future"].done() else None

                if gradcam_img is not None and lime_img is not None:
                    prediction = {
                        "label": st.session_state["pred_label"],
                        "confidence": st.session_state["conf"]
                    }
                    lime_img_uint8 = (lime_img * 255).astype(np.uint8)

                    quick_save(patient_info, img, Image.fromarray(gradcam_img), Image.fromarray(lime_img_uint8), prediction)
                    st.success(f"Saved record for patient `{patient_info['code']}`.")
                else:
                    st.warning("Please wait until both Grad-CAM and LIME finish before saving.")

# ---------------------- TAB 2 ----------------------
with tab2:
    st.title("Records")
    try:
        data = get_patient_records()
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No records found.")
    except Exception as e:
        st.error(f"Error fetching records: {e}")
