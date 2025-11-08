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

# --- Input parameters ---
with st.expander("Input Parameters", expanded=True):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    camh = st.number_input("Enter Camera Height (mm)", value=300)
    ref_h = st.number_input("Enter Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    run_process = st.button("Run Measurement")

# --- Utility functions ---
def centered_visual(img_array, caption=None, width=650):
    if isinstance(img_array, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img_array
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
            <img src="data:image/png;base64,{img_b64}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">
            <div style="text-align:left;width:{width}px;margin-top:6px;">
                <p style="font-size:18px;font-weight:bold;">{caption}</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

def centered_plot(fig, caption=None, width=800):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
            <img src="data:image/png;base64,{img_b64}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">
            <div style="text-align:left;width:{width}px;margin-top:6px;">
                <p style="font-size:18px;font-weight:bold;">{caption}</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# --- Main process ---
if run_process and uploaded_file:
    st.info("Processing image...")

    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_gray = (depth_norm * 255).astype(np.uint8)
    magma = plt.cm.get_cmap("magma")
    depth_color = (magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    if relative_height_ratio == "low": low_bound = 110
    elif relative_height_ratio == "med": low_bound = 100
    elif relative_height_ratio == "high": low_bound = 80
    else: low_bound = 60

    # --- Histogram minima ---
    deriv = np.gradient(smoothed_hist[low_bound:])
    zc = np.where(np.diff(np.sign(deriv)))[0]
    minima_hist = np.array([i for i in zc if deriv[i-1] < 0 and deriv[i+1] > 0]).astype(int) + low_bound
    if minima_hist.size == 0: minima_hist = np.array([int(np.argmin(smoothed_hist))])

    # --- Difference of Gaussians ---
    sigma1, sigma2 = 3.76, 1.8
    sm1 = gaussian_filter1d(hist, sigma=sigma1)
    sm2 = gaussian_filter1d(hist, sigma=sigma2)
    dog = sm1 - sm2
    smooth_dog = gaussian_filter1d(dog, sigma=1.5)

    deriv_dog = np.gradient(smooth_dog)
    zc_dog = np.where(np.diff(np.sign(deriv_dog)))[0]
    maxima_dog = np.array([i for i in zc_dog if deriv_dog[i-1] > 0 and deriv_dog[i+1] < 0]).astype(int)
    minima_dog = np.array([i for i in zc_dog if deriv_dog[i-1] < 0 and deriv_dog[i+1] > 0]).astype(int)

    # --- KMeans midpoint fusion ---
    def kmeans_centers(points, k):
        pts = np.array(points).reshape(-1, 1)
        if pts.shape[0] < k: return np.linspace(0, 255, k)
        km = KMeans(n_clusters=k, random_state=42).fit(pts)
        return np.sort(km.cluster_centers_.ravel())
    c_hist = kmeans_centers(minima_hist, int(nom_of_objects))
    c_dog = kmeans_centers(minima_dog, int(nom_of_objects))
    centers = np.sort((c_hist + c_dog) / 2.0)

    # --- Mask generation ---
    def small_area_remover(binary):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = np.zeros_like(binary)
        if n > 1:
            a = stats[1:, cv2.CC_STAT_AREA]
            out[labels == np.argmax(a)+1] = 255
        return out
    _, ground = cv2.threshold(gray, int(minima_hist[0]), 255, cv2.THRESH_BINARY)
    masks = {}
    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            _, t = cv2.threshold(gray, int(centers[i]), 255, cv2.THRESH_BINARY)
            b = cv2.subtract(ground, t)
            masks[i] = small_area_remover(b)
        total = sum(masks.values())
        r = cv2.subtract(ground, total)
        _, r = cv2.threshold(r, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(r)
    else:
        masks[0] = small_area_remover(ground)
        r = np.zeros_like(gray)

    # --- Measurement utilities ---
    def sad(mask):
        c = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        if c is None: return 0, 0, (0, 0), (0, 0)
        c = np.int32(c)
        x_min, y_min = np.min(c[:, :, 0]), np.min(c[:, :, 1])
        x_max, y_max = np.max(c[:, :, 0]), np.max(c[:, :, 1])
        return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

    def view(dx, dy, px, py, camh=300, f=6.5, vp=[6.144, 8.6], cx=0.82, cy=0.79):
        tx, ty = (dx/px)*vp[1], (dy/py)*vp[0]
        return [(camh/f)*tx*cx, (camh/f)*ty*cy]

    def vertical_text(img, text, org):
        x, y = org; font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), base = cv2.getTextSize(text, font, 1, 3)
        text_img = np.zeros((th+base, tw, 3), np.uint8)
        cv2.putText(text_img, text, (0, th), font, 1, (0,255,0), 3)
        M = cv2.getRotationMatrix2D((tw//2, th//2), 90, 1)
        rot = cv2.warpAffine(text_img, M, (tw, tw), flags=cv2.INTER_LINEAR)
        h, w = rot.shape[:2]
        img[y:y+h, x:x+w] = np.where(rot>0, rot, img[y:y+h, x:x+w])
        return img

    def mean_depth(d, tl, br):
        lx, ly = tl; rx, ry = br
        return np.mean(d[ly:ry, lx:rx])

    # --- Object dimensions ---
    temp = depth_color.copy()
    results, boxes = [], []
    for i in range(nom_of_objects):
        dx, dy, tl, br = sad(masks[i])
        x, y = view(dx, dy, initial_image.shape[0], initial_image.shape[1], camh=camh)
        cv2.rectangle(temp, tl, br, (0,255,0), 2)
        temp = vertical_text(temp, f"Length {int(y)}mm", tl)
        cv2.putText(temp, f"Width {int(x)}mm", (tl[0], br[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        results.append({"Object": i+1, "Width (mm)": int(x), "Length (mm)": int(y)})
        boxes.append([tl, br])

    ref = mean_depth(depth_color, (0,0), boxes[0][0])
    mean_val = [depth_color[(masks[i]//255)==1].mean() for i in range(nom_of_objects)]
    scaler = float(min(mean_val)-ref)
    for i, val in enumerate(mean_val):
        h = ((val-ref)/scaler)*ref_h
        cv2.putText(temp, f"Depth {int(h)}mm", boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
        results[i]["Depth (mm)"] = int(h)

    # --- Display ---
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")
    df = pd.DataFrame(results)
    st.markdown("<h5 style='font-size:18px;'>Object Dimension Measurements</h5>", unsafe_allow_html=True)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size':'14px'}), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("Depth Intensity Histogram", expanded=False):
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(hist, color='gray', alpha=0.6, label="Raw Histogram")
        ax.plot(smoothed_hist, color='orange', linewidth=2, label="Smoothed (σ=1.89)")
        ax.legend(); ax.set_title("Depth Intensity Distribution"); ax.set_xlabel("Intensity"); ax.set_ylabel("Frequency")
        centered_plot(fig, "Figure 5. Raw and smoothed histogram showing intensity distribution.")
        fig_comb, axc = plt.subplots(figsize=(10,4.5))
        axc.plot(3*dog, color='red', label="3× DoG (σ₁=3.76, σ₂=1.8)")
        axc.plot(1.8*smooth_dog, color='green', label="1.8× Smoothed DoG (σ=1.5)")
        axc.scatter(maxima_dog, 1.8*smooth_dog[maxima_dog], color='c', marker='x', s=60, label='Maxima')
        axc.scatter(minima_dog, 1.8*smooth_dog[minima_dog], color='b', marker='x', s=60, label='Minima')
        axc.legend(); axc.set_title("Scaled DoG with Maxima's and Minima's on means:1.8,3.76")
        centered_plot(fig_comb, "Figure 5B. Scaled DoG with maxima and minima (σ₁=3.76, σ₂=1.8, post-smooth σ=1.5).")

    with st.expander("Difference of Gaussians (DoG) Analysis", expanded=False):
        fig_dog, axs = plt.subplots(3,1,figsize=(10,8),sharex=True)
        axs[0].plot(3*dog, 'r'); axs[0].set_title("Raw Scaled DoG (3×) — σ₁=3.76, σ₂=1.8")
        axs[1].plot(1.8*smooth_dog, 'g'); axs[1].set_title("Smoothed Scaled DoG (1.8×, σₛ=1.5)")
        axs[2].plot(3*dog, 'r', alpha=0.6); axs[2].plot(1.8*smooth_dog, 'g')
        axs[2].scatter(maxima_dog, 1.8*smooth_dog[maxima_dog], color='c', marker='x', s=50, label='Maxima')
        axs[2].scatter(minima_dog, 1.8*smooth_dog[minima_dog], color='b', marker='x', s=50, label='Minima')
        axs[2].legend(); axs[2].set_title("Combined Scaled DoG with Maxima & Minima")
        centered_plot(fig_dog, "Figure 5B. Scaled DoG with maxima and minima (σ₁=3.76, σ₂=1.8, post-smooth σ=1.5).")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
