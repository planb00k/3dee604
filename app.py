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

# --- Final fixed vertical text (no clipping) ---
def vertical_text(img, text, org, color=(255, 255, 0), angle=90):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    canvas_h = tw + 40
    canvas_w = th * 3
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    cv2.putText(canvas, text, (10, tw // 2 + th // 2), font, scale, (*color, 255), thick, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((canvas_w // 2, canvas_h // 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    x, y = org
    h, w = rot.shape[:2]
    y1, y2 = max(0, y), min(y + h, img.shape[0])
    x1, x2 = max(0, x), min(x + w, img.shape[1])
    if y1 >= y2 or x1 >= x2:
        return img
    roi = img[y1:y2, x1:x2]
    rot_crop = rot[0:y2 - y1, 0:x2 - x1]
    alpha = rot_crop[:, :, 3:] / 255.0
    img[y1:y2, x1:x2] = (alpha * rot_crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
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
    post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)
    smoothed_hist1 = gaussian_filter1d(hist, sigma=3.76)
    smoothed_hist2 = gaussian_filter1d(hist, sigma=1.8)
    dog = smoothed_hist1 - smoothed_hist2
    smooth_dog = 1.8 * gaussian_filter1d(dog, sigma=1.5)
    low_bound = {"low": 110, "med": 100, "high": 80, "vhigh": 60}[relative_height_ratio]
    upper_bound = 255

    minima_hist_rel = find_local_minima(smoothed_hist[low_bound:upper_bound])
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

    def sad(depthmap, mask):
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

    temp, results, bboxes = depth_color.copy(), [], []
    for i in range(n_clusters):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl, br = sad(temp, mask_i)
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1], camh=camh)
        cv2.rectangle(temp, tl, br, (0, 255, 0), 2)
        bboxes.append([tl, br])

        # --- Length label properly left and visible ---
        box_width = br[0] - tl[0]
        box_height = br[1] - tl[1]
        label_x = tl[0] + int(box_width * 0.1)
        label_y = tl[1] + int(box_height * 0.45)
        temp = vertical_text(temp, f"Length {int(y)} mm", (label_x, label_y), color=(255, 255, 0), angle=90)

        # Width
        cv2.putText(temp, f"Width {int(x)} mm", (tl[0] + 10, br[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        results.append({"Object": i + 1, "Width (mm)": int(x), "Length (mm)": int(y)})

    # Draw Depth last (on top)
    for i, (tl, br) in enumerate(bboxes):
        cv2.putText(temp, f"Depth {ref_h} mm", (br[0] - 140, tl[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Annotated Output")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '16px'}), use_container_width=True)

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
