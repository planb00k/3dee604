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
    nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    run_process = st.button("Run Measurement")

# ---------------- Run Process ----------------
if run_process and uploaded_file:
    st.info("Processing image. Please wait...")

    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))

    # ---------------- Depth Estimation ----------------
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

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

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    # ---------------- Histogram & Derivative ----------------
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    if relative_height_ratio == "low":
        low_bound = 110
    elif relative_height_ratio == "med":
        low_bound = 100
    elif relative_height_ratio == "high":
        low_bound = 80
    else:
        low_bound = 60

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if (i - 1) >= 0 and (i + 1) < len(derivative)
                       and derivative[i - 1] < 0 and derivative[i + 1] > 0]).astype(int) + low_bound
    if minima.size == 0:
        minima = np.array([np.argmin(smoothed_hist)])

    # ---------------- KMeans Segmentation ----------------
    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))

    def small_area_remover(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            output[labels == largest_label] = 255
        return output

    ground_truth = int(minima[0])
    _, ground = cv2.threshold(gray, ground_truth, 255, cv2.THRESH_BINARY)
    masks = {}

    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            _, thresh = cv2.threshold(gray, int(centers[i]), 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)

        sum_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, nom_of_objects):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)
        residual = np.zeros_like(gray)

    # ---------------- Measurement Functions ----------------
    def sad(mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        if corners is None:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return 0, 0, (0, 0), (0, 0)
            return np.ptp(xs), np.ptp(ys), (np.min(xs), np.min(ys)), (np.max(xs), np.max(ys))
        corners = np.int32(corners)
        x_min, y_min = np.min(corners[:, :, 0]), np.min(corners[:, :, 1])
        x_max, y_max = np.max(corners[:, :, 0]), np.max(corners[:, :, 1])
        return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        tx = (dx / px) * viewport[1]
        ty = (dy / py) * viewport[0]
        x = (camh / f) * tx
        y = (camh / f) * ty
        return [cx * x, cy * y]

    def vertical_text(img, text, org):
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1, 3
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (0, text_h), font, scale, (0, 255, 0), thickness)
        M = cv2.getRotationMatrix2D((text_w // 2, text_h // 2), 90, 1.0)
        nW, nH = int((text_h + text_w)), int((text_h + text_w))
        M[0, 2] += (nW / 2) - text_w // 2
        M[1, 2] += (nH / 2) - text_h // 2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        h, w = rotated.shape[:2]
        if y + h <= img.shape[0] and x + w <= img.shape[1]:
            img[y:y+h, x:x+w] = np.where(rotated > 0, rotated, img[y:y+h, x:x+w])
        return img

    def mean_depth(depthmap, lt_p, rb_p):
        lx, ly = lt_p; rx, ry = rb_p
        ly, ry, lx, rx = max(0, ly), min(depthmap.shape[0], ry), max(0, lx), min(depthmap.shape[1], rx)
        return np.mean(depthmap[ly:ry, lx:rx])

    # ---------------- Measurement and Annotation ----------------
    temp = depth_color.copy()
    bounding_boxes, results = [], []

    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(masks[i])
        x_mm, y_mm = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                          f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        cv2.putText(temp, f"Width {int(x_mm)}mm", (tl_p[0], br_p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        temp = vertical_text(temp, f"Length {int(y_mm)}mm", tl_p)
        results.append({"Object": i+1, "Width (mm)": int(x_mm), "Length (mm)": int(y_mm)})

    ref = mean_depth(depth_color, (0,0), bounding_boxes[0][0])
    mean_val, min1 = [], 255
    for i in range(nom_of_objects):
        _01img = masks[i]//255
        meanint = depth_color[_01img==1].mean() if np.any(_01img==1) else depth_color.mean()
        if ref < meanint < min1: min1 = meanint
        mean_val.append(meanint)
    scaler = (min1 - ref) if (min1 - ref)!=0 else 1.0

    for i in range(nom_of_objects):
        temph = ((mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm", bounding_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        results[i]["Depth (mm)"] = int(temph)

    # ---------------- Helper Display Functions ----------------
    def centered_visual(img, caption):
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:40px">
            <img src="data:image/png;base64,{img_b64}" width="550">
            <p style="text-align:left;font-size:18px;font-weight:bold;">{caption}</p>
        </div>
        """, unsafe_allow_html=True)

    def centered_plot(fig, caption):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:40px">
            <img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" width="550">
            <p style="text-align:left;font-size:18px;font-weight:bold;">{caption}</p>
        </div>
        """, unsafe_allow_html=True)
        plt.close(fig)

    # ---------------- Display ----------------
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size':'14px'}), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("Depth Intensity Histogram", expanded=False):
        fig_hist, ax = plt.subplots(figsize=(6,3))
        ax.plot(hist, color='gray', alpha=0.6, label="Raw Histogram")
        ax.plot(smoothed_hist, color='orange', linewidth=2, label="Gaussian Smoothed Histogram")
        ax.legend(); ax.set_title("Depth Intensity Distribution")
        centered_plot(fig_hist, "Figure 5. Raw and smoothed histogram.")

        # ✅ Fixed DoG computation (from smoothed histogram)
        g1 = gaussian_filter1d(smoothed_hist, sigma=1.0)
        g2 = gaussian_filter1d(smoothed_hist, sigma=3.0)
        display_dog = g1 - g2

        fig_comb, axc = plt.subplots(figsize=(6,3))
        axc.plot(hist, color='blue', alpha=0.6, label="Raw Histogram")
        axc.plot(smoothed_hist, color='orange', linewidth=2, label="Smoothed Histogram")
        axc.plot(display_dog, color='red', linewidth=1.5, label="DoG (σ₁=1, σ₂=3)")
        axc.legend()
        centered_plot(fig_comb, "Figure 5B. Combined histogram and DoG overlay (computed from smoothed histogram).")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
