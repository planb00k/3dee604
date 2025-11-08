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

# ---------------- Helper Functions ----------------
def small_area_remover(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

def find_local_minima(arr):
    g = np.gradient(arr)
    return np.where((np.concatenate(([g[0]], g[:-1])) < 0) & (g > 0))[0]

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
    return np.sort(kmeans.cluster_centers_.reshape(-1))

def centered_visual(img_array, caption=None, width=550):
    if isinstance(img_array, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img_array
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px;border-radius:6px;">
        <div style="text-align:left;width:{width}px;margin-top:6px;">
            <p style="font-size:16px;font-weight:600;">{caption or ''}</p>
        </div>
    </div>""", unsafe_allow_html=True)

def centered_plot(fig, caption, width=700):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px;border-radius:6px;">
        <div style="text-align:left;width:{width}px;margin-top:6px;">
            <p style="font-size:16px;font-weight:600;">{caption or ''}</p>
        </div>
    </div>""", unsafe_allow_html=True)

@st.cache_resource
def load_depth_model():
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    return processor, model

# ---- vertical text (rotated single-line, always visible)
def vertical_text(img, text, org, color=(255, 255, 0), angle=90):
    """Draw rotated text that stays visible using alpha blending."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    canvas = np.zeros((th + 20, tw + 20, 4), dtype=np.uint8)
    # black outline
    cv2.putText(canvas, text, (10, th + 10), font, scale, (0, 0, 0, 255), thick + 2, cv2.LINE_AA)
    # main text
    cv2.putText(canvas, text, (10, th + 10), font, scale, (*color, 255), thick, cv2.LINE_AA)
    # rotate
    M = cv2.getRotationMatrix2D((tw // 2, th // 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (tw, th), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    rgb, a = rot[:, :, :3], rot[:, :, 3] / 255.0
    x, y = org
    h, w = rgb.shape[:2]
    y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    roi = img[y:y2, x:x2]
    a = a[:y2 - y, :x2 - x][..., None]
    roi[:] = (a * rgb[:y2 - y, :x2 - x] + (1 - a) * roi).astype(np.uint8)
    img[y:y2, x:x2] = roi
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
    pp = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = pp[0].get("predicted_depth", pp[0].get("depth")).squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.cvtColor((plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    sm1, sm2 = gaussian_filter1d(hist, 3.76), gaussian_filter1d(hist, 1.8)
    dog = sm1 - sm2
    smooth = 1.8 * gaussian_filter1d(dog, 1.5)
    low = {"low":110, "med":100, "high":80, "vhigh":60}[relative_height_ratio]
    up = 255
    mh = sm1[low:up]
    min_hist = (find_local_minima(mh) + low).astype(int)
    grad = np.gradient(smooth)
    zc = np.where(np.diff(np.sign(grad)))[0]
    min_dog = np.array([i for i in zc if grad[i - 1] < 0 and grad[i + 1] > 0], int)
    min_dog = min_dog[(min_dog >= low) & (min_dog < up)]

    n = int(max(1, nom_of_objects))
    ch, cd = safe_kmeans_centers(min_hist, n, low, up), safe_kmeans_centers(min_dog, n, low, up)
    cm = np.sort((np.array(ch) + np.array(cd)) / 2.0).astype(int)

    masks, thr = {}, int(cm[0]) if len(cm) else low
    _, ground = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    if n > 1:
        for i in range(1, n):
            thr = int(cm[i]) if i < len(cm) else int(cm[-1])
            _, t = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            m = cv2.subtract(ground, t)
            masks[i] = small_area_remover(m)
        s = np.zeros_like(gray, np.uint8)
        for i in range(1, n): s = cv2.add(s, masks[i])
        res = cv2.subtract(ground, s)
        _, res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(res)
    else:
        masks[0], res = small_area_remover(ground), np.zeros_like(gray)

    def sad(mask):
        if mask is None or np.count_nonzero(mask) == 0:
            h, w = gray.shape
            return w, h, (0, 0), (w - 1, h - 1)
        c = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        if c is None:
            h, w = gray.shape
            return w, h, (0, 0), (w - 1, h - 1)
        c = np.int32(c)
        x0, y0, x1, y1 = np.min(c[:, :, 0]), np.min(c[:, :, 1]), np.max(c[:, :, 0]), np.max(c[:, :, 1])
        return x1 - x0, y1 - y0, (x0, y0), (x1, y1)

    def view(dx, dy, px, py, camh=300, f=5.42, vp=[6.144, 8.6], cx=0.82, cy=0.79):
        return [cx * (camh / f) * (dx / px) * vp[1],
                cy * (camh / f) * (dy / py) * vp[0]]

    def mean_depth(dm, lt, rb):
        lx, ly = lt; rx, ry = rb
        lx, rx = max(0, lx), min(dm.shape[1]-1, rx)
        ly, ry = max(0, ly), min(dm.shape[0]-1, ry)
        return float(dm.mean()) if ry <= ly or rx <= lx else np.mean(dm[ly:ry, lx:rx])

    temp, res_tab, boxes = depth_color.copy(), [], []
    for i in range(n):
        dx, dy, tl, br = sad(masks.get(i))
        x, y = view(dx, dy, initial_image.shape[0], initial_image.shape[1], camh)
        cv2.rectangle(temp, tl, br, (0, 255, 0), 2)
        boxes.append([tl, br])
        temp = vertical_text(temp, f"Length {int(y)}mm", (tl[0]+10, tl[1]+40), (255,255,0), 90)
        cv2.putText(temp, f"Width {int(x)}mm", (tl[0]+10, br[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        res_tab.append({"Object": i+1, "Width (mm)": int(x), "Length (mm)": int(y)})

    ref = mean_depth(depth_color, (0,0), boxes[0][0])
    mvals, min1 = [], 255
    for i in range(n):
        m = masks[i]//255
        val = depth_color[m==1].mean() if np.count_nonzero(m) else float(depth_color.mean())
        if ref < val < min1: min1 = val
        mvals.append(val)
    scale = float(min1 - ref) if (min1 - ref)!=0 else 1.0
    for i in range(n):
        d = (float(mvals[i]-ref)/scale)*ref_h
        cv2.putText(temp, f"Depth {int(d)}mm", boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
        res_tab[i]["Depth (mm)"] = int(d)

    # ---- Display ----
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing Width, Length, and Depth values.")
    df = pd.DataFrame(res_tab)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size':'16px'}), use_container_width=True)
else:
    if run_process:
        st.warning("Please upload an image before running the measurement.")
