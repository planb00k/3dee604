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
from typing import Tuple, List

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("3D Object Measurement (Width, Length, Depth)")

# -------------------------
# Helper / UI Components
# -------------------------
def img_to_base64(img_bgr: np.ndarray, width: int) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">'

def centered_visual(img_array, caption: str = None, width: int = 650):
    if isinstance(img_array, np.ndarray):
        html_img = img_to_base64(img_array, width)
    else:
        buf = io.BytesIO()
        img_array.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        html_img = f'<img src="data:image/png;base64,{data}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">'
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:36px;">
        {html_img}
        <div style="text-align:left;width:{width}px;margin-top:6px;">
            <p style="font-size:18px;font-weight:bold;margin:0;padding:0;">{caption or ""}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def centered_plot(fig: plt.Figure, caption: str = None, width: int = 800):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    html_img = f'<img src="data:image/png;base64,{data}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">'
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:36px;">
        {html_img}
        <div style="text-align:left;width:{width}px;margin-top:6px;">
            <p style="font-size:18px;font-weight:bold;margin:0;padding:0;">{caption or ""}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Model Loading (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_depth_model(model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    return processor, model, device

# -------------------------
# Core Helper Functions
# -------------------------
def run_kmeans_safe(points: List[int], k: int) -> np.ndarray:
    pts = np.array(points).reshape(-1, 1).astype(float)
    if pts.shape[0] < k:
        return np.sort(np.linspace(0, 255, k))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pts)
    return np.sort(kmeans.cluster_centers_.reshape(-1))

def small_area_remover(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

def sad(mask: np.ndarray) -> Tuple[int, int, Tuple[int,int], Tuple[int,int]]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(mask_u8, 10, 0.05, 50)
    if corners is None:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0, (0, 0), (0, 0)
        return int(np.ptp(xs)), int(np.ptp(ys)), (int(np.min(xs)), int(np.min(ys))), (int(np.max(xs)), int(np.max(ys)))
    corners = np.int32(corners)
    x_min, y_min = int(np.min(corners[:, :, 0])), int(np.min(corners[:, :, 1]))
    x_max, y_max = int(np.max(corners[:, :, 0])), int(np.max(corners[:, :, 1]))
    return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

def view(dx: float, dy: float, px: int, py: int, camh: float = 300.0, f: float = 6.5, viewport: List[float] = [6.144, 8.6], cx: float = 0.82, cy: float = 0.79) -> Tuple[float,float]:
    tx = (dx / px) * viewport[1]
    ty = (dy / py) * viewport[0]
    x = (camh / f) * tx
    y = (camh / f) * ty
    return (cx * x, cy * y)

def vertical_text(img: np.ndarray, text: str, org: Tuple[int,int], font_scale: float = 1, thickness: int = 3, angle: int = 90) -> np.ndarray:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
    cv2.putText(text_img, text, (0, text_h), font, font_scale, (0, 255, 0), thickness)
    M = cv2.getRotationMatrix2D((text_w // 2, text_h // 2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW = int((text_h * sin) + (text_w * cos))
    nH = int((text_h * cos) + (text_w * sin))
    M[0, 2] += (nW / 2) - text_w // 2
    M[1, 2] += (nH / 2) - text_h // 2
    rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    h, w = rotated.shape[:2]
    place_x = x - w - 6
    place_y = y
    place_x = max(0, place_x)
    place_y = max(0, place_y)
    roi = img[place_y:place_y + h, place_x:place_x + w]
    if roi.shape[0] == h and roi.shape[1] == w:
        mask_rot = (rotated > 0)
        img[place_y:place_y + h, place_x:place_x + w] = np.where(mask_rot, rotated, roi)
    return img

def mean_depth(depthmap: np.ndarray, lt_p: Tuple[int,int], rb_p: Tuple[int,int]) -> float:
    lx, ly = lt_p; rx, ry = rb_p
    ly = max(0, ly); ry = min(depthmap.shape[0], ry)
    lx = max(0, lx); rx = min(depthmap.shape[1], rx)
    if ry <= ly or rx <= lx:
        return float(depthmap.mean())
    return float(np.mean(depthmap[ly:ry, lx:rx]))

# -------------------------
# Input Panel
# -------------------------
with st.expander("Input Parameters", expanded=True):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    camh = st.number_input("Enter Camera Height (mm)", value=300)
    ref_h = st.number_input("Enter Reference Object Height (mm)", value=50)
    num_objects = st.number_input("Number of Objects", value=1, min_value=1, step=1)
    run_process = st.button("Run Measurement")

# -------------------------
# Main Pipeline
# -------------------------
if run_process and uploaded_file:
    st.info("Processing image...")

    image = Image.open(uploaded_file).convert("RGB")
    initial_image = np.array(image)

    processor, model, device = load_depth_model()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    magma = plt.cm.get_cmap("magma")
    depth_color = (magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    # -------------------------
    # Adaptive DoG extrema detection
    # -------------------------
    sigma1, sigma2 = 3.76, 1.8
    sm1, sm2 = gaussian_filter1d(hist, sigma=sigma1), gaussian_filter1d(hist, sigma=sigma2)
    dog = sm1 - sm2
    smooth_dog = gaussian_filter1d(dog, sigma=1.5)

    hist_energy = hist / np.sum(hist)
    cum_energy = np.cumsum(hist_energy)
    low_bound = np.searchsorted(cum_energy, 0.05)
    upper_bound = np.searchsorted(cum_energy, 0.95)
    low_bound, upper_bound = max(20, low_bound), min(250, upper_bound)

    derivative_dog = np.gradient(smooth_dog[low_bound:upper_bound])
    zero_crossings = np.where(np.diff(np.sign(derivative_dog)))[0]
    maxima_dog = np.array([i for i in zero_crossings if derivative_dog[i - 1] > 0 and derivative_dog[i + 1] < 0]).astype(int) + low_bound
    minima_dog = np.array([i for i in zero_crossings if derivative_dog[i - 1] < 0 and derivative_dog[i + 1] > 0]).astype(int) + low_bound

    amplitude_threshold = np.max(np.abs(smooth_dog)) * 0.05
    minima_dog = np.array([m for m in minima_dog if abs(smooth_dog[m]) > amplitude_threshold])
    maxima_dog = np.array([m for m in maxima_dog if abs(smooth_dog[m]) > amplitude_threshold])

    if minima_dog.size == 0:
        minima_dog = np.array([int(np.argmin(smooth_dog))])
    if maxima_dog.size == 0:
        maxima_dog = np.array([int(np.argmax(smooth_dog))])

    # -------------------------
    # Threshold Clustering & Masking
    # -------------------------
    centers_hist = run_kmeans_safe(minima_dog, int(num_objects))
    centers_dog = run_kmeans_safe(minima_dog, int(num_objects))
    centers = np.sort(((centers_hist + centers_dog) / 2.0).reshape(-1))

    ground_val = int(minima_dog[0])
    _, ground = cv2.threshold(gray, ground_val, 255, cv2.THRESH_BINARY)
    masks = {}

    if num_objects > 1:
        for i in range(1, int(num_objects)):
            thr = int(centers[i]) if i < len(centers) else int(centers[-1])
            _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)
        sum_mask = np.zeros_like(gray)
        for i in range(1, int(num_objects)):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)
        residual = np.zeros_like(gray)

    # -------------------------
    # Measurement Computation
    # -------------------------
    temp = depth_color.copy()
    bounding_boxes, results = [], []

    for i in range(int(num_objects)):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(mask_i)
        x_mm, y_mm = view(dx, dy, px=image.height, py=image.width, camh=float(camh))
        if tl_p != (0, 0):
            cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
            cv2.putText(temp, f"Width {int(x_mm)}mm", (tl_p[0], br_p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            temp = vertical_text(temp, f"Length {int(y_mm)}mm", (tl_p[0], tl_p[1]))
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x_mm), "Length (mm)": int(y_mm)})

    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val = [float(depth_color[(masks[i] // 255) == 1].mean()) if np.any(masks[i]) else float(depth_color.mean()) for i in range(int(num_objects))]
    min1 = min([m for m in mean_val if m > ref], default=ref + 1)
    scaler = (min1 - ref) if (min1 - ref) != 0 else 1.0

    for i in range(int(num_objects)):
        temph = (mean_val[i] - ref) / scaler * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm", bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

    # -------------------------
    # Display
    # -------------------------
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '14px'}), use_container_width=True)

    # DoG Visualization
    with st.expander("DoG Analysis", expanded=False):
        fig_comb, ax_comb = plt.subplots(figsize=(10, 4.5))
        ax_comb.plot(3 * dog, color='red', label="3× DoG (σ₁=3.76, σ₂=1.8)")
        ax_comb.plot(1.8 * smooth_dog, color='green', label="1.8× Smoothed DoG (σ=1.5)")
        ax_comb.scatter(maxima_dog, (1.8 * smooth_dog)[maxima_dog], color='cyan', marker='x', s=60, label='Maxima')
        ax_comb.scatter(minima_dog, (1.8 * smooth_dog)[minima_dog], color='blue', marker='x', s=60, label='Minima')
        ax_comb.axvline(x=low_bound, color='gray', linestyle='--', label=f"low={low_bound}")
        ax_comb.axvline(x=upper_bound, color='gray', linestyle='--', label=f"up={upper_bound}")
        ax_comb.set_title(f"Scaled DoG with Maxima's and Minima's (σ₁={sigma1}, σ₂={sigma2})")
        ax_comb.set_xlabel("Pixel Intensity")
        ax_comb.set_ylabel("Value")
        ax_comb.legend()
        centered_plot(fig_comb, "Figure 5B. Adaptive Scaled DoG with maxima & minima, cropped by energy bounds.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
else:
    st.info("Upload an image and set parameters, then press 'Run Measurement'.")
