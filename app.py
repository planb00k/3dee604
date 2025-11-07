import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("📏 3D Object Measurement (Length, Width & Depth)")

# ---------------------- USER INPUTS ----------------------
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"], index=1)
    camh = st.number_input("Camera Height (mm)", value=300)
    ref_h = st.number_input("Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    run = st.button("Run Analysis")

# ---------------------- MAIN PROCESS ----------------------
if uploaded_file and run:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # ----------- DEPTH ESTIMATION -----------
    model_id = "Intel/dpt-hybrid-midas"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
    depth = cv2.resize(predicted_depth, (img_rgb.shape[1], img_rgb.shape[0]))

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_RGB2GRAY)

    # ----------- HISTOGRAM & MINIMA -----------
    hist = cv2.calcHist([depth_gray], [0], None, [256], [0, 256]).flatten()
    sigma = 1.89
    smoothed = gaussian_filter1d(hist, sigma=sigma)

    if relative_height_ratio == "low":
        low_bound, error_rct = 110, 1.08
    elif relative_height_ratio == "med":
        low_bound, error_rct = 100, 1.23
    elif relative_height_ratio == "high":
        low_bound, error_rct = 80, 2.91
    else:
        low_bound, error_rct = 60, 3.0

    derivative = np.gradient(smoothed[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound

    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))

    # ----------- THRESHOLDING / MASKS -----------
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
        for i in range(1, nom_of_objects):
            _, thr = cv2.threshold(depth_gray, centers[i], 255, cv2.THRESH_BINARY)
            binary = ground - thr
            masks[i] = small_area_remover(binary)
        sum_mask = np.zeros_like(depth_gray, dtype=np.uint8)
        for i in range(1, nom_of_objects):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)

    # ----------- WIDTH / LENGTH COMPUTATION -----------
    def sad(mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        corners = np.int32(corners)
        x_min, y_min = np.min(corners[:,:,0]), np.min(corners[:,:,1])
        x_max, y_max = np.max(corners[:,:,0]), np.max(corners[:,:,1])
        return [x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)]

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144,8.6], cx=0.82, cy=0.79):
        tx, ty = (dx/px)*viewport[1], (dy/py)*viewport[0]
        x, y = (camh/f)*tx, (camh/f)*ty
        return [cx*x, cy*y]

    def mean_depth_val(depth, tl, br):
        lx, ly = tl
        rx, ry = br
        return np.mean(depth[ly:ry, lx:rx])

    annotated = depth_color.copy()
    boxes, mean_vals = [], []

    for i in range(nom_of_objects):
        dx, dy, tl, br = sad(masks[i])
        width_mm, length_mm = view(dx, dy, px=img_rgb.shape[0], py=img_rgb.shape[1], camh=camh)
        cv2.rectangle(annotated, tl, br, (0,255,0), 2)
        cv2.putText(annotated, f"W={int(width_mm)}mm", (tl[0], br[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(annotated, f"L={int(length_mm)}mm", (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        boxes.append([tl, br])
        mean_vals.append(mean_depth_val(depth, tl, br))

    # ----------- DEPTH COMPUTATION -----------
    ref = np.mean(depth[:boxes[0][0][1], :boxes[0][0][0]])
    min_depth = min(mean_vals)
    scaler = abs(min_depth - ref)

    for i in range(nom_of_objects):
        calc_depth = abs((mean_vals[i] - ref) / scaler) * ref_h
        tl, _ = boxes[i]
        cv2.putText(annotated, f"D={int(calc_depth)}mm", (tl[0], tl[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # ----------- VISUALS -----------
    st.subheader("Main Visuals")
    c1, c2 = st.columns(2)
    c1.image(depth_color, caption="Colorized Depth Map", use_column_width=True)
    c2.image(ground, caption="Thresholded Binary Segmentation", use_column_width=True)

    c3, c4 = st.columns(2)
    mask_vis = np.zeros_like(depth_color)
    for i, m in masks.items():
        color = np.random.randint(0, 255, size=3)
        mask_vis[m > 0] = color
    c3.image(mask_vis, caption="Individual Object Masks", use_column_width=True)
    c4.image(annotated, caption="Final Annotated Image", use_column_width=True)

    # ----------- ANALYSIS PLOTS -----------
    st.subheader("Analysis Plots")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(hist, color='gray', label='Original Hist')
    ax[0].plot(smoothed, color='red', label='Smoothed')
    ax[0].set_title("Histogram & Gaussian Smooth")
    ax[0].legend()

    dog = smoothed - gaussian_filter1d(smoothed, sigma=3)
    ax[1].plot(dog, color='blue')
    ax[1].set_title("Difference of Gaussian (DoG)")

    indices_bar = np.tile(np.arange(256), (20, 1))
    cmap_img = plt.cm.magma(indices_bar / 255.0)
    for c in centers:
        ax[2].axvline(x=c, color='lime', linestyle='--', label='Object minima')
    ax[2].imshow(cmap_img, extent=[0, 256, 0, 1], aspect='auto')
    ax[2].set_title("Located Indices (Magma colormap)")
    st.pyplot(fig)

    # ----------- LOGS -----------
    with st.expander("📜 Internal Logs"):
        st.write({
            "Minima Detected": minima.tolist(),
            "Cluster Centers": centers.tolist(),
            "Scaler": scaler,
            "Reference Depth": ref,
            "Mean Depths": mean_vals
        })
