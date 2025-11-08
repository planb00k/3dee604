# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import base64
import io

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("3D Object Measurement (Width, Length, Depth)")

# ---------------- Input Section ----------------
with st.expander("Input Parameters", expanded=True):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    camh = st.number_input("Enter Camera Height (mm)", value=300)
    ref_h = st.number_input("Enter Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=2, min_value=1)
    run_process = st.button("Run Measurement")

# ---------------- Helpers ----------------
def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

def find_local_minima(arr):
    g = np.gradient(arr)
    minima_idx = np.where((np.concatenate(([g[0]], g[:-1])) < 0) & (g > 0))[0]
    return minima_idx

def safe_kmeans_centers(points, n_clusters, low=0, high=255):
    if points is None or len(points) == 0:
        return np.linspace(low, high, n_clusters + 1)[1:]
    pts = np.array(points).reshape(-1, 1).astype(float)
    if pts.shape[0] < n_clusters:
        extra_needed = n_clusters - pts.shape[0]
        extra = np.linspace(low, high, extra_needed + 2, dtype=int)[1:-1]
        if extra.size > 0:
            pts = np.vstack([pts, extra.reshape(-1, 1).astype(float)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pts)
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))
    return centers

def centered_visual(img_array, caption=None, width=550):
    if isinstance(img_array, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img_array
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; margin-bottom:40px;">
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px; border-radius:6px;">
        <div style="text-align:left; width:{width}px; margin-top:6px;">
            <p style="font-size:16px; font-weight:600;">{caption or ''}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def centered_plot(fig, caption, width=700):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; margin-bottom:40px;">
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px; border-radius:6px;">
        <div style="text-align:left; width:{width}px; margin-top:6px;">
            <p style="font-size:16px; font-weight:600;">{caption or ''}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

@st.cache_resource
def load_depth_model():
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    return processor, model

# ---------------- Run Process ----------------
if run_process and uploaded_file:
    st.info("Processing image. Please wait...")

    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))

    processor, model = load_depth_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    depth_result = post_processed[0]

    if "predicted_depth" in depth_result:
        depth = depth_result["predicted_depth"].squeeze().cpu().numpy()
    elif "depth" in depth_result:
        depth = depth_result["depth"].squeeze().cpu().numpy()
    else:
        raise KeyError(f"Depth key missing: {depth_result.keys()}")

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    # ---------------- Histogram & DoG ----------------
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    sigma1, sigma2 = 3.76, 1.8
    smoothed_hist1 = gaussian_filter1d(hist, sigma=sigma1)
    smoothed_hist2 = gaussian_filter1d(hist, sigma=sigma2)
    dog = smoothed_hist1 - smoothed_hist2
    smooth_dog = 1.8 * gaussian_filter1d(dog, sigma=1.5)
    scaled_dog = 3 * (smoothed_hist1 - smoothed_hist2)

    if relative_height_ratio == "low":
        low_bound = 110
    elif relative_height_ratio == "med":
        low_bound = 100
    elif relative_height_ratio == "high":
        low_bound = 80
    else:
        low_bound = 60
    upper_bound = 255

    mh_window = smoothed_hist[low_bound:upper_bound]
    dog_window = smooth_dog[low_bound:upper_bound]
    minima_hist_rel = find_local_minima(mh_window)
    minima_hist = (minima_hist_rel + low_bound).astype(int)

    n_clusters = int(max(1, nom_of_objects))
    centers_hist = safe_kmeans_centers(minima_hist, n_clusters, low=low_bound, high=upper_bound)
    centers_dog = safe_kmeans_centers(minima_hist, n_clusters, low=low_bound, high=upper_bound)
    centers_mid = np.sort((centers_hist + centers_dog) / 2.0).astype(int)

    # segmentation
    masks = {}
    ground_threshold = int(centers_mid[0]) if len(centers_mid) > 0 else low_bound
    _, ground = cv2.threshold(gray, ground_threshold, 255, cv2.THRESH_BINARY)

    if n_clusters > 1:
        for i in range(1, n_clusters):
            thr_val = int(centers_mid[i]) if i < len(centers_mid) else int(centers_mid[-1])
            _, thresh = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)
        sum_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, n_clusters):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)
        residual = np.zeros_like(gray)

    # ---------------- Measurement ----------------
    def sad(camheight, depthmap, mask):
        try:
            if mask is None or np.count_nonzero(mask) == 0:
                h, w = depthmap.shape[:2]
                return w, h, (0, 0), (w - 1, h - 1)
            corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
            if corners is None:
                h, w = depthmap.shape[:2]
                return w, h, (0, 0), (w - 1, h - 1)
            corners = np.int32(corners)
            x_min = np.min(corners[:, :, 0])
            y_min = np.min(corners[:, :, 1])
            x_max = np.max(corners[:, :, 0])
            y_max = np.max(corners[:, :, 1])
            return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)
        except Exception:
            h, w = depthmap.shape[:2]
            return w, h, (0, 0), (w - 1, h - 1)

    def view(dx, dy, px, py, camh=300, f=5.42, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        tx = (dx / px) * viewport[1]
        ty = (dy / py) * viewport[0]
        x = (camh / f) * tx
        y = (camh / f) * ty
        return [cx * x, cy * y]

    def mean_depth(depth_map, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        lx = max(0, min(depth_map.shape[1]-1, lx))
        rx = max(0, min(depth_map.shape[1]-1, rx))
        ly = max(0, min(depth_map.shape[0]-1, ly))
        ry = max(0, min(depth_map.shape[0]-1, ry))
        if ry <= ly or rx <= lx:
            return float(depth_map.mean())
        return np.mean(depth_map[ly:ry, lx:rx])

    temp = depth_color.copy()
    bounding_boxes, results = [], []
    for i in range(n_clusters):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=mask_i)
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1], f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x), "Length (mm)": int(y)})

    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val, min1 = [], 255
    for i in range(n_clusters):
        _01img = masks[i] // 255
        meanint = depth_color[_01img == 1].mean() if np.count_nonzero(_01img) else depth_color.mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0
    for i in range(n_clusters):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        results[i]["Depth (mm)"] = int(temph)

    # ---------------- Display ----------------
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index'), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("DoG (Report-style) Visualization", expanded=False):
        fig_dog, ax_dog = plt.subplots(figsize=(10, 4))
        ax_dog.plot(scaled_dog, color='red', label='3×(Gσ1 - Gσ2)')
        ax_dog.plot(smooth_dog, color='green', label='1.8×Smoothed DoG (σ=1.5)')

        # --- FIX: detect minima directly on smooth_dog ---
        grad = np.gradient(smooth_dog)
        zero_crossings = np.where(np.diff(np.sign(grad)))[0]
        minima_indices = [i for i in zero_crossings if grad[i - 1] < 0 and grad[i + 1] > 0]
        minima_indices = np.clip(np.array(minima_indices, dtype=int), 0, len(smooth_dog) - 1)

        if len(minima_indices) > 0:
            ax_dog.scatter(minima_indices, smooth_dog[minima_indices], c='b', marker='x', s=40, label='Minima (DoG)')

        if minima_hist.size > 0:
            mh = np.clip(minima_hist.astype(int), 0, len(smoothed_hist) - 1)
            ax_dog.scatter(mh, smoothed_hist[mh], c='c', marker='x', s=40, label='Minima (Smoothed Hist)', zorder=5)

        for cm in centers_mid:
            ax_dog.axvline(x=int(cm), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        ax_dog.set_title("Scaled DoG with Minima (DoG & Smoothed Histogram) and Midpoints")
        ax_dog.set_xlabel("Intensity bins (0–255)")
        ax_dog.set_ylabel("Amplitude")
        ax_dog.legend()
        ax_dog.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        centered_plot(fig_dog, "Figure 6. Corrected DoG minima aligned with the green curve (true zero-crossings).")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
