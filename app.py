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

def img_to_base64(img_bgr, width):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    html = f'<img src="data:image/png;base64,{data}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">'
    return html

def centered_visual(img_array, caption=None, width=650):
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
            <p style="font-size:18px;font-weight:bold;margin:0;padding:0;">{caption}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def centered_plot(fig, caption=None, width=800):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    html_img = f'<img src="data:image/png;base64,{data}" style="display:block;margin:0 auto;width:{width}px;border-radius:6px;">'
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:36px;">
        {html_img}
        <div style="text-align:left;width:{width}px;margin-top:6px;">
            <p style="font-size:18px;font-weight:bold;margin:0;padding:0;">{caption}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

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
    depth_result = post_processed[0]
    if "predicted_depth" in depth_result:
        depth = depth_result["predicted_depth"].squeeze().cpu().numpy()
    elif "depth" in depth_result:
        depth = depth_result["depth"].squeeze().cpu().numpy()
    else:
        raise KeyError(f"Depth key missing: {depth_result.keys()}")

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_gray = (depth_norm * 255).astype(np.uint8)
    magma = plt.cm.get_cmap("magma")
    depth_color = (magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

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

    deriv_hist = np.gradient(smoothed_hist[low_bound:])
    zc_hist = np.where(np.diff(np.sign(deriv_hist)))[0]
    minima_hist = np.array([i for i in zc_hist if (i - 1) >= 0 and (i + 1) < len(deriv_hist) and deriv_hist[i - 1] < 0 and deriv_hist[i + 1] > 0]).astype(int) + low_bound
    if minima_hist.size == 0:
        minima_hist = np.array([int(np.argmin(smoothed_hist))])

    sigma1 = 3.76
    sigma2 = 1.8
    sm1 = gaussian_filter1d(hist, sigma=sigma1)
    sm2 = gaussian_filter1d(hist, sigma=sigma2)
    dog = sm1 - sm2
    smooth_dog = gaussian_filter1d(dog, sigma=1.5)

    deriv_dog = np.gradient(smooth_dog)
    zc_dog = np.where(np.diff(np.sign(deriv_dog)))[0]
    maxima_dog = np.array([i for i in zc_dog if deriv_dog[i - 1] > 0 and deriv_dog[i + 1] < 0]).astype(int)
    minima_dog = np.array([i for i in zc_dog if deriv_dog[i - 1] < 0 and deriv_dog[i + 1] > 0]).astype(int)
    if minima_dog.size == 0:
        minima_dog = np.array([int(np.argmin(smooth_dog))])
    if maxima_dog.size == 0:
        maxima_dog = np.array([int(np.argmax(smooth_dog))])

    def run_kmeans_safe(points, k):
        pts = np.array(points).reshape(-1, 1).astype(float)
        if pts.shape[0] < k:
            return np.sort(np.linspace(0, 255, k))
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pts)
        return np.sort(kmeans.cluster_centers_.reshape(-1))

    centers_hist = run_kmeans_safe(minima_hist, int(nom_of_objects))
    centers_dog = run_kmeans_safe(minima_dog, int(nom_of_objects))
    centers = np.sort(((centers_hist + centers_dog) / 2.0).reshape(-1))

    def small_area_remover(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            output[labels == largest_label] = 255
        return output

    ground_val = int(minima_hist[0])
    _, ground = cv2.threshold(gray, ground_val, 255, cv2.THRESH_BINARY)
    masks = {}
    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            thr = int(centers[i])
            _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
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

    def vertical_text(img, text, org, font_scale=1, thickness=3, angle=90):
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
        if place_x < 0:
            place_x = x + 6
        if place_y + h > img.shape[0]:
            place_y = max(0, img.shape[0] - h - 6)
        roi = img[place_y:place_y + h, place_x:place_x + w]
        if roi.shape[0] == h and roi.shape[1] == w:
            img[place_y:place_y + h, place_x:place_x + w] = np.where(rotated > 0, rotated, roi)
        return img

    def mean_depth(depthmap, lt_p, rb_p):
        lx, ly = lt_p; rx, ry = rb_p
        ly = max(0, ly); ry = min(depthmap.shape[0], ry)
        lx = max(0, lx); rx = min(depthmap.shape[1], rx)
        if ry <= ly or rx <= lx:
            return float(depthmap.mean())
        return float(np.mean(depthmap[ly:ry, lx:rx]))

    temp = depth_color.copy()
    bounding_boxes = []
    results = []

    for i in range(nom_of_objects):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(mask_i)
        x_mm, y_mm = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1], camh=camh)
        cv2.circle(temp, tl_p, 5, (0, 255, 0), 2)
        cv2.circle(temp, br_p, 5, (0, 255, 0), 2)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x_mm), "Length (mm)": int(y_mm)})
        cv2.putText(temp, f"Width {int(x_mm)}mm", (tl_p[0], br_p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        temp = vertical_text(temp, f"Length {int(y_mm)}mm", (tl_p[0], tl_p[1]))

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

    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

    bbox_only = depth_color.copy()
    for i, (tl, br) in enumerate(bounding_boxes):
        cv2.rectangle(bbox_only, tl, br, (0, 255, 0), 2)
        cv2.putText(bbox_only, f"{i+1}", (tl[0]-28, tl[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
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
        ax_hist.plot(smoothed_hist, label="Gaussian Smoothed Histogram (σ=1.89)", color='orange', linewidth=2)
        ax_hist.set_title("Depth Intensity Distribution")
        ax_hist.set_xlabel("Pixel Intensity (0–255)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        centered_plot(fig_hist, "Figure 5. Raw and smoothed histogram showing the depth intensity distribution.")

        fig_comb, ax_comb = plt.subplots(figsize=(10, 4.5))
        ax_comb.plot(3 * dog, color='red', label="3× DoG (σ₁=3.76, σ₂=1.8)")
        ax_comb.plot(1.8 * smooth_dog, color='green', label="1.8× Smoothed DoG (σ=1.5)")
        ax_comb.scatter(maxima_dog, (1.8 * smooth_dog)[maxima_dog], marker='x', color='c', s=60, label='Maxima (DoG)', zorder=5)
        ax_comb.scatter(minima_dog, (1.8 * smooth_dog)[minima_dog], marker='x', color='b', s=60, label='Minima (DoG)', zorder=6)
        ax_comb.set_title("Scaled DoG with Maxima's and Minima's on means:1.8,3.76")
        ax_comb.set_xlabel("Pixel Intensity")
        ax_comb.set_ylabel("Value")
        ax_comb.legend()
        centered_plot(fig_comb, "Figure 5B. Scaled DoG with maxima and minima (σ₁ = 3.76, σ₂ = 1.8, post-smooth σ = 1.5).")

    with st.expander("Difference of Gaussians (DoG) Analysis", expanded=False):
        fig_dog, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(3 * dog, color='red', linewidth=1.2)
        axs[0].set_title("Raw Scaled DoG (3×) — σ₁ = 3.76, σ₂ = 1.8")
        axs[0].set_ylabel("DoG Value")
        axs[1].plot(1.8 * smooth_dog, color='green', linewidth=1.5)
        axs[1].set_title("Smoothed Scaled DoG (1.8×, σ = 1.5)")
        axs[1].set_ylabel("Smoothed Value")
        axs[2].plot(3 * dog, color='red', alpha=0.6, linewidth=1.0, label="3×(G₍3.76₎−G₍1.8₎)")
        axs[2].plot(1.8 * smooth_dog, color='green', linewidth=1.5, label="1.8×Smoothed DoG (σ=1.5)")
        axs[2].scatter(maxima_dog, (1.8 * smooth_dog)[maxima_dog], color='cyan', marker='x', s=50, label='Maxima')
        axs[2].scatter(minima_dog, (1.8 * smooth_dog)[minima_dog], color='blue', marker='x', s=50, label='Minima')
        axs[2].axhline(0, color='gray', linestyle='--', linewidth=1)
        axs[2].set_title("Combined Scaled DoG with Maxima & Minima")
        axs[2].set_xlabel("Pixel Intensity (0–255)")
        axs[2].set_ylabel("Value")
        axs[2].legend(fontsize=9, loc='upper right')
        centered_plot(fig_dog, "Figure 5B. Scaled DoG with maxima and minima (σ₁ = 3.76, σ₂ = 1.8, post-smooth σ = 1.5).")

    with st.expander("KMeans Threshold Fusion (Histogram + DoG)", expanded=False):
        fig_km_hist, ax_km_hist = plt.subplots(figsize=(6, 3))
        ax_km_hist.plot(smoothed_hist, color='black', label="Smoothed Histogram")
        colors = ['blue', 'green', 'purple', 'brown', 'magenta']
        for idx, c in enumerate(centers_hist):
            color = colors[idx % len(colors)]
            ax_km_hist.axvline(x=c, color=color, linestyle='--', linewidth=1.5, label=f"Hist Center {idx + 1} ({int(c)})")
            ax_km_hist.text(int(c) + 3, max(smoothed_hist) * 0.05, f"H{idx+1}", color=color, fontsize=10)
        ax_km_hist.set_title("KMeans on Histogram Minima")
        ax_km_hist.set_xlabel("Pixel Intensity")
        ax_km_hist.set_ylabel("Smoothed Frequency")
        ax_km_hist.legend(fontsize=9)
        centered_plot(fig_km_hist, "Figure 7. KMeans clustering applied to histogram minima for segmentation threshold selection.")

        fig_km_dog, ax_km_dog = plt.subplots(figsize=(6, 3))
        ax_km_dog.plot(smooth_dog, color='orange', label="Smoothed DoG")
        for idx, c in enumerate(centers_dog):
            color = colors[idx % len(colors)]
            ax_km_dog.axvline(x=c, color=color, linestyle='--', linewidth=1.5, label=f"DoG Center {idx + 1} ({int(c)})")
            ax_km_dog.text(int(c) + 3, max(smooth_dog) * 0.05, f"D{idx+1}", color=color, fontsize=10)
        ax_km_dog.set_title("KMeans on DoG Minima")
        ax_km_dog.set_xlabel("Pixel Intensity")
        ax_km_dog.set_ylabel("DoG Value")
        ax_km_dog.legend(fontsize=9)
        centered_plot(fig_km_dog, "Figure 7B. KMeans clustering applied to smoothed DoG minima.")

        fig_km_fused, ax_km_fused = plt.subplots(figsize=(6, 3))
        ax_km_fused.plot(smoothed_hist, color='black', label="Smoothed Histogram")
        for idx, c in enumerate(centers):
            color = colors[idx % len(colors)]
            ax_km_fused.axvline(x=c, color=color, linestyle='--', linewidth=1.5, label=f"Fused Center {idx + 1} ({int(c)})")
            ax_km_fused.text(int(c) + 3, max(smoothed_hist) * 0.05, f"F{idx+1}", color=color, fontsize=10)
        ax_km_fused.set_title("Fused Midpoint Thresholds (Histogram + DoG)")
        ax_km_fused.set_xlabel("Pixel Intensity")
        ax_km_fused.set_ylabel("Smoothed Frequency")
        ax_km_fused.legend(fontsize=9)
        centered_plot(fig_km_fused, "Figure 7C. Element-wise midpoint between histogram and DoG cluster centers used as final thresholds.")

    with st.expander("Segmentation and Object Masks", expanded=False):
        lap = cv2.filter2D(depth_color, -1, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
        edges = cv2.Canny(depth_color, 100, 200)
        centered_visual(lap, "Figure 8. Laplacian filtered depth color image.")
        centered_visual(edges, "Figure 9. Canny edge map of depth color image.")
        centered_visual(ground, "Figure 10. Ground threshold mask after initial binary segmentation.")
        for key, mask in sorted(masks.items(), key=lambda x: x[0]):
            centered_visual(mask, f"Figure 11.{key + 1} Object Mask {key + 1} after area refinement using connected components.")
        centered_visual(residual, "Figure 12. Residual mask showing unassigned or background regions after segmentation.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.") 
