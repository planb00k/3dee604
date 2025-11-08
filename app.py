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
import base64, io

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

# ---------------- Helper Functions ----------------
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


# -------- FIXED vertical text (no clipping of "mm") ----------
def vertical_text(img, text, org, color=(255, 255, 0), angle=90):
    """Draw vertical text with full visibility and anti-clipping."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    # create large enough transparent canvas
    pad = 150
    canvas = np.zeros((tw + pad * 2, th + pad * 4, 4), dtype=np.uint8)

    org_x = pad
    org_y = (canvas.shape[0] + th) // 2
    cv2.putText(canvas, text, (org_x, org_y), font, scale, (*color, 255), thick, cv2.LINE_AA)

    # Rotate safely with padding
    M = cv2.getRotationMatrix2D((canvas.shape[1] // 2, canvas.shape[0] // 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (canvas.shape[1] * 2, canvas.shape[0] * 2),
                         flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    x, y = org
    h, w = rot.shape[:2]
    y = max(0, min(y, img.shape[0] - h))
    x = max(0, min(x, img.shape[1] - w))

    alpha = rot[:, :, 3:] / 255.0
    roi = img[y:y + h, x:x + w]
    blended = (alpha * rot[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    img[y:y + h, x:x + w] = blended
    return img


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
    depth = depth_result.get("predicted_depth", depth_result.get("depth")).squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    # ---- Histogram & DoG ----
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)
    smoothed_hist1 = gaussian_filter1d(hist, sigma=3.76)
    smoothed_hist2 = gaussian_filter1d(hist, sigma=1.8)
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
    minima_hist_rel = find_local_minima(mh_window)
    minima_hist = (minima_hist_rel + low_bound).astype(int) if minima_hist_rel.size > 0 else np.array([], dtype=int)

    grad = np.gradient(smooth_dog)
    zero_crossings = np.where(np.diff(np.sign(grad)))[0]
    minima_dog = np.array([i for i in zero_crossings if grad[i - 1] < 0 and grad[i + 1] > 0], dtype=int)
    minima_dog = minima_dog[(minima_dog >= low_bound) & (minima_dog < upper_bound)]

    n_clusters = int(max(1, nom_of_objects))
    centers_hist = safe_kmeans_centers(minima_hist, n_clusters, low=low_bound, high=upper_bound)
    centers_dog = safe_kmeans_centers(minima_dog, n_clusters, low=low_bound, high=upper_bound)
    centers_mid = np.sort((np.array(centers_hist) + np.array(centers_dog)) / 2.0).astype(int)

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

    # ---- Measurement ----
    def sad(camheight, depthmap, mask):
        if mask is None or np.count_nonzero(mask) == 0:
            h, w = depthmap.shape[:2]
            return w, h, (0, 0), (w - 1, h - 1)
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        if corners is None:
            h, w = depthmap.shape[:2]
            return w, h, (0, 0), (w - 1, h - 1)
        corners = np.int32(corners)
        x_min, y_min = np.min(corners[:, :, 0]), np.min(corners[:, :, 1])
        x_max, y_max = np.max(corners[:, :, 0]), np.max(corners[:, :, 1])
        return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

    def view(dx, dy, px, py, camh=300, f=5.42, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        tx, ty = (dx / px) * viewport[1], (dy / py) * viewport[0]
        return [cx * (camh / f) * tx, cy * (camh / f) * ty]

    def mean_depth(depth_map, lt, rb):
        lx, ly = lt
        rx, ry = rb
        lx, rx = max(0, lx), min(depth_map.shape[1]-1, rx)
        ly, ry = max(0, ly), min(depth_map.shape[0]-1, ry)
        if ry <= ly or rx <= lx:
            return float(depth_map.mean())
        return np.mean(depth_map[ly:ry, lx:rx])

    temp, results, bboxes = depth_color.copy(), [], []
    for i in range(n_clusters):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl, br = sad(camheight=camh, depthmap=temp, mask=mask_i)
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1], camh=camh)
        cv2.rectangle(temp, tl, br, (0, 255, 0), 2)
        bboxes.append([tl, br])

        center_x = (tl[0] + br[0]) // 2
        img_center = initial_image.shape[1] // 2
        if center_x < img_center:
            label_x, label_angle = br[0] + 15, 90
        else:
            label_x, label_angle = tl[0] - 60, 270

        temp = vertical_text(
            temp,
            f"Length {int(y)} mm",
            (label_x, tl[1] + 20),
            color=(255, 255, 0),
            angle=label_angle
        )

        cv2.putText(temp, f"Width {int(x)} mm", (tl[0] + 10, br[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        results.append({"Object": i + 1, "Width (mm)": int(x), "Length (mm)": int(y)})

    ref = mean_depth(depth_color, (0, 0), bboxes[0][0])
    mean_val, min1 = [], 255
    for i in range(n_clusters):
        _01img = masks[i] // 255
        meanint = depth_color[_01img == 1].mean() if np.count_nonzero(_01img) else float(depth_color.mean())
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0

    for i in range(n_clusters):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)} mm", bboxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

    # ---- Display ----
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing Width, Length, and Depth values.")

    bbox_only = depth_color.copy()
    for i, (tl, br) in enumerate(bboxes):
        cv2.rectangle(bbox_only, tl, br, (0, 255, 0), 2)
        cv2.putText(bbox_only, f"Obj {i+1}", (tl[0], br[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    centered_visual(bbox_only, "Figure 1B. Detected object bounding boxes before annotation.")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '16px'}), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("Original and Depth Representations", expanded=False):
        centered_visual(initial_image, "Figure 2. Original RGB image.")
        centered_visual(depth_gray, "Figure 3. Grayscale depth map.")
        centered_visual(depth_color, "Figure 4. Colorized depth map (magma colormap).")

    with st.expander("Depth Intensity Histogram & Smoothed", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(hist, label="Raw Histogram", alpha=0.6)
        ax.plot(smoothed_hist, label="Gaussian Smoothed (σ=1.89)", color='red', linewidth=2)
        ax.legend()
        centered_plot(fig, "Figure 5. Raw and smoothed histogram.")

    with st.expander("DoG Visualization", expanded=False):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(scaled_dog, color='red', label='3×(Gσ1 - Gσ2)')
        ax.plot(smooth_dog, color='green', label='1.8×Smoothed DoG (σ=1.5)')
        if len(minima_dog) > 0:
            md = np.clip(minima_dog.astype(int), 0, len(smooth_dog) - 1)
            ax.scatter(md, smooth_dog[md], c='b', marker='x', s=40, label='Minima (DoG)', zorder=5)
        if len(minima_hist) > 0:
            mh = np.clip(minima_hist.astype(int), 0, len(smooth_dog) - 1)
            ax.scatter(mh, smooth_dog[mh], c='c', marker='x', s=40, label='Minima (Smoothed Hist)', zorder=5)
        for cm in centers_mid:
            ax.axvline(x=int(cm), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title("Scaled DoG with Minima (DoG & Smoothed Histogram) and Midpoints")
        ax.set_xlabel("Intensity bins (0–255)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        centered_plot(fig, "Figure 6. All minima markers aligned on DoG curve.")

    with st.expander("Segmentation and Object Masks", expanded=False):
        centered_visual(ground, "Figure 7. Ground threshold mask.")
        for key, mask in sorted(masks.items(), key=lambda x: x[0]):
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            centered_visual(mask_bgr, f"Figure 8.{key+1} Object Mask {key+1}.")
        centered_visual(residual, "Figure 9. Residual/background mask.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
