import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
CNN_DIR = ROOT / "models" / "cnn"
UNET_DIR = ROOT / "models" / "unet"

st.set_page_config(page_title="BrainScanAI", page_icon="🧠", layout="wide")
st.title("🧠 BrainScanAI")
st.caption("Experimental educational prototype — not a medical diagnosis.")

with st.sidebar:
    st.header("Model status")
    st.write("CNN:", "configured" if list(CNN_DIR.glob("*.keras")) else "model file missing")
    st.write("U-Net:", "configured" if list(UNET_DIR.glob("*.keras")) else "model file missing")

def _dice_metric(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reduce_mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))


def _dice_bce(y_true, y_pred):
    import tensorflow as tf
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1.0 - _dice_metric(y_true, y_pred))


# Custom objects used when loading U-Net artifacts saved with a dice loss/metric.
UNET_CUSTOM_OBJECTS = {
    "dice_metric": _dice_metric,
    "dice_bce": _dice_bce,
    "dice_coefficient": _dice_metric,
    "dice_loss": _dice_bce,
}

@st.cache_resource(show_spinner="Loading CNN model...")
def load_cnn(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False, custom_objects=UNET_CUSTOM_OBJECTS)

@st.cache_resource(show_spinner="Loading U-Net model...")
def load_unet(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False, custom_objects=UNET_CUSTOM_OBJECTS)

def pick_model(folder):
    # Prefer the most recently added model so a freshly trained artifact wins.
    files = sorted(folder.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None

def infer_input_shape(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    # Expected channels-last image model; use model's own spatial dimensions if fixed.
    h = int(shape[1]) if len(shape) >= 4 and shape[1] else 224
    w = int(shape[2]) if len(shape) >= 4 and shape[2] else 224
    c = int(shape[3]) if len(shape) >= 4 and shape[3] else 3
    return h, w, c

def prepare_cnn_image(image, model):
    from PIL import Image
    h, w, c = infer_input_shape(model)
    if c == 1:
        img = image.convert("L").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)[..., None]
    else:
        img = image.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    # This CNN is EfficientNetB0-based and already contains its own input
    # preprocessing (a Rescaling 1/255 layer followed by the ImageNet
    # Normalization layer). It therefore expects raw pixel values in [0, 255].
    # Do NOT divide by 255 here: doing so feeds the model values near 0.004 and
    # collapses every prediction onto a single class. Verified against the
    # bundled model on labeled data: raw [0,255] -> ~98% accuracy, /255 -> ~25%.
    return np.expand_dims(arr, 0)

def show_image(image, **kwargs):
    """st.image wrapper that works on Streamlit <1.38 and >=1.38."""
    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, use_column_width=True, **kwargs)

def predict_cnn(image):
    model_path = pick_model(CNN_DIR)
    if not model_path:
        raise FileNotFoundError("Place your CNN .keras file in models/cnn/.")
    model = load_cnn(model_path)
    x = prepare_cnn_image(image, model)
    pred = np.asarray(model.predict(x, verbose=0))
    pred = np.squeeze(pred)
    # Output order of the bundled CNN, verified by running it on labeled
    # samples: index 0 = glioma, 1 = meningioma, 2 = no tumor, 3 = pituitary.
    classes = ["glioma", "meningioma", "no tumor", "pituitary"]
    if pred.ndim == 0:
        return [{"class": "model output", "score": float(pred)}]
    if pred.ndim != 1:
        pred = pred.reshape(-1)
    # If the model emits logits rather than probabilities, convert to softmax.
    if np.any(pred < 0) or not np.isclose(np.sum(pred), 1.0, atol=0.05):
        exp = np.exp(pred - np.max(pred))
        pred = exp / np.sum(exp)
    labels = classes if len(pred) == len(classes) else [f"class_{i}" for i in range(len(pred))]
    order = np.argsort(pred)[::-1]
    return [{"class": labels[i], "score": float(pred[i])} for i in order]

def predict_unet(image):
    model_path = pick_model(UNET_DIR)
    if not model_path:
        raise FileNotFoundError("Place your trained U-Net .keras file in models/unet/.")
    model = load_unet(model_path)
    # Match the spatial size the U-Net was built with (64/128/256...).
    h, w, _ = infer_input_shape(model)
    gray = image.convert("L").resize((w, h), Image.BILINEAR)
    # Training preprocessing: grayscale, resized, scaled to [0, 1]. The model ends
    # in a sigmoid, so its output is already a per-pixel probability map.
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    output = np.asarray(model.predict(arr[None, ..., None], verbose=0))
    prob = np.squeeze(output)
    if prob.ndim != 2:
        raise ValueError(f"Unexpected U-Net output shape: {output.shape}")
    binary = prob >= 0.5
    mask_img = Image.fromarray((binary.astype(np.uint8) * 255))
    # Red overlay so the predicted region is visible on top of the MRI.
    base = np.asarray(gray.convert("RGB"), dtype=np.uint8).copy()
    base[binary] = (0.45 * base[binary] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    return {"mask": mask_img, "overlay": Image.fromarray(base), "fraction": float(binary.mean())}

st.subheader("Upload an MRI image")
uploaded = st.file_uploader("Choose a brain MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="mri")
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    # Identify the current upload so a stale CNN result from a previous image
    # cannot gate the U-Net for this one.
    file_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    left, right = st.columns(2)
    with left:
        show_image(image, caption="Uploaded image")
    with right:
        if st.button("Run CNN classification", type="primary"):
            try:
                results = predict_cnn(image)
                st.session_state["last_prediction"] = results
                st.session_state["last_prediction_id"] = file_id
                st.subheader("Experimental CNN output")
                for item in results:
                    st.write(f"**{item['class']}** — {item['score']:.1%}")
                st.caption("Scores are model outputs, not calibrated medical probabilities.")
            except Exception as exc:
                st.error(f"CNN could not run: {exc}")
        # Only honour a CNN result that belongs to the image currently uploaded.
        last = st.session_state.get("last_prediction")
        if st.session_state.get("last_prediction_id") != file_id:
            last = None
        top_class = last[0]["class"] if last else None
        is_no_tumor = top_class == "no tumor"
        # If the CNN says there is no tumor, do not segment or show any mask.
        if is_no_tumor:
            st.info(
                "CNN classified this slice as **no tumor**, so U-Net segmentation is "
                "skipped. Upload a different slice or re-run the CNN to segment."
            )
        if st.button("Run U-Net segmentation", disabled=is_no_tumor):
            try:
                seg = predict_unet(image)
                show_image(seg["overlay"], caption="Predicted tumor region (red) over the MRI")
                st.image(seg["mask"], caption="Predicted binary mask", clamp=True)
                st.write(f"Predicted mask coverage: {seg['fraction']:.1%} of this resized slice.")
                if seg["fraction"] <= 0.0005:
                    st.info("The U-Net predicted an empty mask for this slice.")
                st.warning("Segmentation is experimental. Verify preprocessing and masks before interpreting results.")
            except Exception as exc:
                st.error(f"U-Net could not run: {exc}")
else:
    st.info("Upload a single MRI slice. This prototype is not designed to process a complete 3D MRI volume.")
