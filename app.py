# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("📏 3D Object Measurement (Width, Length, Depth)")

# --- Upload + Input Sidebar ---
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    camh = st.number_input("Camera Height (mm)", value=300)
    ref_h = st.number_input("Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    run_button = st.button("Run Pipeline 🚀")

if uploaded_file and run_button:
    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))
    img_rgb = initial_image.copy()
    img = Image.fromarray(img_rgb)

    # --- Depth Estimation ---
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)]
    )

    depth_result = post_processed[0]
    if "predicted_depth" in depth_result:
        depth = depth_result["predicted_depth"].squeeze().cpu().numpy()
    elif "depth" in depth_result:
        depth = depth_result["depth"].squeeze().cpu().numpy()
    else:
        raise KeyError(f"Depth key missing: {depth_result.keys()}")

    # --- Normalize and Colorize ---
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
    depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)

    # --- Histogram & Smoothing ---
    hist = cv2.calcHist([depth_gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    if relative_height_ratio == "low":
        low_bound = 110
    elif relative_height_ratio == "med":
        low_bound = 100
    elif relative_height_ratio == "high":
        low_bound = 80
    elif relative_height_ratio == "vhigh":
        low_bound = 60

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([
        i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0
    ]).astype(int) + low_bound

    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.reshape(len(kmeans.cluster_centers_)))

    # --- MASK GENERATION (fixed overlap issue) ---
    def small_area_remover(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = np.argmax(areas) + 1
            output[labels == largest] = 255
        return output

    ground_thr = minima[0]
    _, ground = cv2.threshold(depth_gray, ground_thr, 255, cv2.THRESH_BINARY)

    masks = {}
    if nom_of_objects > 1:
        previous = np.zeros_like(depth_gray)
        for i in range(nom_of_objects):
            threshold_val = centers[i] if i < len(centers) else centers[-1]
            _, thresh = cv2.threshold(depth_gray, threshold_val, 255, cv2.THRESH_BINARY)
            mask = cv2.subtract(ground, thresh)
            mask = cv2.subtract(mask, previous)
            mask = small_area_remover(mask)
            masks[i] = mask
            previous = cv2.add(previous, mask)
    else:
        masks[0] = small_area_remover(ground)

    # --- Calculation functions ---
    def sad(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return [0, 0, (0, 0), (0, 0)]
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return [w, h, (x, y), (x + w, y + h)]

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        tx, ty = (dx / px) * viewport[1], (dy / py) * viewport[0]
        x, y = (camh / f) * tx, (camh / f) * ty
        return [cx * x, cy * y]

    def mean_depth_val(depth, mask):
        y, x = np.where(mask > 0)
        return np.mean(depth[y, x]) if len(y) > 0 else 0

    # --- Annotate results ---
    annotated = depth_color.copy()
    boxes, mean_vals = [], []

    for i in range(nom_of_objects):
        dx, dy, tl, br = sad(masks[i])
        width_mm, length_mm = view(dx, dy, px=img_rgb.shape[0], py=img_rgb.shape[1], camh=camh)
        cv2.rectangle(annotated, tl, br, (0, 255, 0), 2)
        cv2.putText(annotated, f"W={int(width_mm)}mm", (tl[0], br[1] + 25*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"L={int(length_mm)}mm", (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        boxes.append([tl, br])
        mean_vals.append(mean_depth_val(depth, masks[i]))

    ref = np.mean(depth[:boxes[0][0][1], :boxes[0][0][0]])
    scaler = abs(max(mean_vals) - min(mean_vals)) + 1e-6

    for i in range(nom_of_objects):
        calc_depth = abs((mean_vals[i] - ref) / scaler) * ref_h
        tl, _ = boxes[i]
        cv2.putText(annotated, f"D={int(calc_depth)}mm", (tl[0], tl[1] + 50 + 25*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- Visual outputs ---
    st.subheader("Main Visuals")
    col1, col2 = st.columns(2)
    col1.image(depth_color, caption="Colorized Depth Map", use_column_width=True)
    col2.image(ground, caption="Thresholded Binary Segmentation", use_column_width=True)

    st.image(annotated, caption="Final Annotated Image", use_column_width=True)

    st.subheader("Individual Object Masks")
    mask_cols = st.columns(min(3, nom_of_objects))
    for i in range(nom_of_objects):
        mask_cols[i % 3].image(masks[i], caption=f"Mask {i+1}", use_column_width=True)

    # --- Histogram visualization ---
    st.subheader("Analysis Plots")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hist, label="Original Hist", alpha=0.4)
    ax.plot(smoothed_hist, label="Gaussian Smoothed", color="orange")
    for m in minima:
        ax.axvline(x=m, color="red", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_title("Histogram & Minima Detection")
    st.pyplot(fig)

    st.success("✅ Processing Complete!")
