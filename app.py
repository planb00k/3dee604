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

    # ---------------- Histogram & Derivative (used for segmentation) ----------------
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

    # --- Original gradient logic (kept for segmentation stability) ---
    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([
        i for i in zero_crossings
        if (i - 1) >= 0 and (i + 1) < len(derivative) and derivative[i - 1] < 0 and derivative[i + 1] > 0
    ]).astype(int) + low_bound

    if minima.size == 0:
        minima = np.array([np.argmin(smoothed_hist)])


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
            thr_val = int(centers[i])
            _, thresh = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
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

    
    def sad(mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        if corners is None:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return 0, 0, (0, 0), (0, 0)
            return int(np.ptp(xs)), int(np.ptp(ys)), (int(np.min(xs)), int(np.min(ys))), (int(np.max(xs)), int(np.max(ys)))
        corners = np.int32(corners)
        x_min, y_min = int(np.min(corners[:, :, 0])), int(np.min(corners[:, :, 1]))
        x_max, y_max = int(np.max(corners[:, :, 0])), int(np.max(corners[:, :, 1]))
        return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        tx = (dx / px) * viewport[1]
        ty = (dy / py) * viewport[0]
        x = (camh / f) * tx
        y = (camh / f) * ty
        return [cx * x, cy * y]

    # === ORIGINAL vertical_text behavior restored (top-left origin placement) ===
    def vertical_text(img, text, org):
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 3
        angle = 90
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (0, text_h), font, scale, (0, 255, 0), thickness)
        M = cv2.getRotationMatrix2D((text_w // 2, text_h // 2), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int((text_h * sin) + (text_w * cos))
        nH = int((text_h * cos) + (text_w * sin))
        M[0, 2] += (nW / 2) - text_w // 2
        M[1, 2] += (nH / 2) - text_h // 2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        h, w = rotated.shape[:2]
        if y + h <= img.shape[0] and x + w <= img.shape[1]:
            img[y:y + h, x:x + w] = np.where(rotated > 0, rotated, img[y:y + h, x:x + w])
        return img

    def mean_depth(depthmap, lt_p, rb_p):
        lx, ly = lt_p; rx, ry = rb_p
        ly, ry, lx, rx = max(0, ly), min(depthmap.shape[0], ry), max(0, lx), min(depthmap.shape[1], rx)
        if ry <= ly or rx <= lx:
            return float(depthmap.mean())
        return float(np.mean(depthmap[ly:ry, lx:rx]))

    # ---------------- Measurement and Annotation ----------------
    temp = depth_color.copy()
    bounding_boxes = []
    results = []

    for i in range(nom_of_objects):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(mask_i)
        # ensure ints
        tl_p = (int(tl_p[0]), int(tl_p[1])); br_p = (int(br_p[0]), int(br_p[1]))
        x_mm, y_mm = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                          f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.circle(temp, tl_p, 5, (0, 255, 0), 2)
        cv2.circle(temp, br_p, 5, (0, 255, 0), 2)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x_mm), "Length (mm)": int(y_mm)})
        # width text (below)
        cv2.putText(temp, f"Width {int(x_mm)}mm", (tl_p[0], br_p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # vertical length text — ORIGINAL placement (start at top-left tl_p)
        temp = vertical_text(temp, f"Length {int(y_mm)}mm", tl_p)

    # compute depth values
    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val = []
    min1 = 255
    for i in range(nom_of_objects):
        _01img = (masks[i] // 255) if (i in masks) else np.zeros_like(gray)
        if np.any(_01img == 1):
            meanint = float(depth_color[_01img == 1].mean())
        else:
            meanint = float(depth_color.mean())
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0

    for i in range(nom_of_objects):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm", bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

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
            <img src="data:image/png;base64,{img_b64}" style="display:block; margin:0 auto; width:{width}px; border-radius:6px;">
            <div style="text-align:left; width:{width}px; margin-top:6px;">
                <p style="font-size:18px; font-weight:bold;">{caption}</p>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    def centered_plot(fig, caption=None, width=550):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        html = f"""
        <div style="display:flex; flex-direction:column; align-items:center; margin-bottom:40px;">
            <img src="data:image/png;base64,{img_b64}" style="display:block; margin:0 auto; width:{width}px; border-radius:6px;">
            <div style="text-align:left; width:{width}px; margin-top:6px;">
                <p style="font-size:18px; font-weight:bold;">{caption}</p>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

  
    bbox_only = depth_color.copy()
    for i, (tl, br) in enumerate(bounding_boxes):
        cv2.rectangle(bbox_only, tl, br, (0, 255, 0), 2)
        cv2.putText(bbox_only, f"Obj {i+1}", (tl[0], br[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    centered_visual(bbox_only, "Figure 1B. Detected object bounding boxes before dimension annotation.")

  
    df = pd.DataFrame(results)
    st.markdown("<h5 style='font-size:18px;'>Object Dimension Measurements</h5>", unsafe_allow_html=True)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '14px'}), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("Original and Depth Representations", expanded=False):
        centered_visual(initial_image, "Figure 2. Original RGB image used for depth analysis.")
        centered_visual(depth_gray, "Figure 3. Grayscale depth map representing normalized pixel depth values.")
        centered_visual(depth_color, "Figure 4. Colorized depth map using magma colormap for visualizing relative distances.")

    with st.expander("Depth Intensity Histogram", expanded=False):
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
        ax_hist.plot(hist, label="Raw Histogram", alpha=0.6, color='gray')
        ax_hist.plot(smoothed_hist, label="Gaussian Smoothed Histogram", color='orange', linewidth=2)
        ax_hist.set_title("Depth Intensity Distribution")
        ax_hist.set_xlabel("Pixel Intensity (0–255)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        centered_plot(fig_hist, "Figure 5. Raw and smoothed histogram showing the depth intensity distribution.")

        g1 = gaussian_filter1d(smoothed_hist, sigma=1.0)
        g2 = gaussian_filter1d(smoothed_hist, sigma=3.0)
        display_dog = g1 - g2

        fig_comb, ax_comb = plt.subplots(figsize=(6, 3))
        ax_comb.plot(hist, color='blue', alpha=0.6, label="Raw Histogram")
        ax_comb.plot(smoothed_hist, color='orange', linewidth=2, label="Smoothed Histogram")
        ax_comb.plot(display_dog, color='red', linewidth=1.5, label="DoG (σ₁=1, σ₂=3)")
        ax_comb.set_title("Combined Histogram and DoG Overlay")
        ax_comb.set_xlabel("Pixel Intensity (0–255)")
        ax_comb.set_ylabel("Frequency / DoG Value")
        ax_comb.legend()
        centered_plot(fig_comb, "Figure 5B. Combined plot showing raw and smoothed histograms with the Difference of Gaussians (DoG) overlay (computed from smoothed histogram).")

    with st.expander("Difference of Gaussians (DoG) Analysis", expanded=False):
        fig_dog, ax_dog = plt.subplots(figsize=(6, 3))
        ax_dog.plot(display_dog, color='orange', linewidth=1.5, label="Difference of Gaussians (DoG)")
        ax_dog.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_dog.set_title("Difference of Gaussians (DoG) – Bandpass Feature Extraction")
        ax_dog.set_xlabel("Pixel Intensity (0–255)")
        ax_dog.set_ylabel("DoG Value")
        ax_dog.legend()
        centered_plot(fig_dog, "Figure 6. Difference of Gaussians (DoG) curve (visual-only).")

        fig_min, ax_min = plt.subplots(figsize=(6, 3))
        ax_min.plot(smoothed_hist, color='black', linewidth=2)
        ax_min.scatter(minima, smoothed_hist[minima], color='red', s=40, label="Detected Minima")
        for i, m in enumerate(minima):
            ax_min.text(m, smoothed_hist[m] + max(smoothed_hist) * 0.03, f"{i+1}", color='red', ha='center', fontsize=10)
        ax_min.set_title("Detected Minima on Smoothed Histogram")
        ax_min.set_xlabel("Pixel Intensity")
        ax_min.set_ylabel("Smoothed Frequency")
        ax_min.legend()
        centered_plot(fig_min, "Figure 6B. Located minima points on smoothed histogram used for segmentation threshold estimation.")

    with st.expander("KMeans Clustering Overview", expanded=False):
        fig_km, ax_km = plt.subplots(figsize=(6, 3))
        ax_km.plot(smoothed_hist, color='black', label="Smoothed Histogram")
        colors = ['blue', 'green', 'purple', 'brown', 'magenta']
        for idx, c in enumerate(centers):
            color = colors[idx % len(colors)]
            ax_km.axvline(x=c, color=color, linestyle='--', linewidth=1.5, label=f"Cluster Center {idx + 1} (Intensity={int(c)})")
            ax_km.text(int(c) + 3, max(smoothed_hist) * 0.05, f"C{idx+1}", color=color, fontsize=10)
        ax_km.set_title("Cluster-Based Threshold Identification with Labeled Centers")
        ax_km.set_xlabel("Pixel Intensity")
        ax_km.set_ylabel("Smoothed Frequency")
        ax_km.legend(fontsize=9)
        centered_plot(fig_km, "Figure 7. KMeans clustering applied to histogram minima for segmentation threshold selection.")

    with st.expander("Segmentation and Object Masks", expanded=False):
        centered_visual(ground, "Figure 8. Ground threshold mask after initial binary segmentation.")
        for key, mask in sorted(masks.items(), key=lambda x: x[0]):
            centered_visual(mask, f"Figure 9.{key + 1} Object Mask {key + 1} after area refinement using connected components.")
        centered_visual(residual, "Figure 10. Residual mask showing unassigned or background regions after segmentation.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
