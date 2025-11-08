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

with st.expander("Input Parameters", expanded=True):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    camh = st.number_input("Enter Camera Height (mm)", value=300)
    ref_h = st.number_input("Enter Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    run_process = st.button("Run Measurement")

def centered_visual(img_array, caption=None, width=550):
    if isinstance(img_array, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img_array
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
      <img src="data:image/png;base64,{img_b64}" style="width:{width}px;border-radius:6px;">
      <div style="text-align:left;width:{width}px;margin-top:6px;">
        <p style="font-size:18px;font-weight:bold;">{caption}</p>
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

def centered_plot(fig, caption=None, width=650):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:40px;">
      <img src="data:image/png;base64,{img_b64}" style="width:{width}px;border-radius:6px;">
      <div style="text-align:left;width:{width}px;margin-top:6px;">
        <p style="font-size:18px;font-weight:bold;">{caption}</p>
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

if run_process and uploaded_file:
    st.info("Processing image. Please wait...")

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
    depth = depth_result["predicted_depth"].squeeze().cpu().numpy()

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

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array(
        [i for i in zero_crossings if derivative[i - 1] < 0 and derivative[i + 1] > 0]
    ).astype(int) + low_bound
    if minima.size == 0:
        minima = np.array([int(np.argmin(smoothed_hist))])

    # ----------- Correct DoG (Teammate Version) -----------
    raw_hist = hist.copy()
    sigma1, sigma2, sigma_smooth = 3.76, 1.8, 1.5
    gauss1 = gaussian_filter1d(raw_hist, sigma=sigma1)
    gauss2 = gaussian_filter1d(raw_hist, sigma=sigma2)
    dog_raw = gauss1 - gauss2
    dog_scaled = 3 * dog_raw
    dog_smooth = gaussian_filter1d(dog_raw, sigma=sigma_smooth)
    dog_smooth_scaled = 1.8 * dog_smooth

    derivative_dog = np.gradient(dog_smooth)
    zc_dog = np.where(np.diff(np.sign(derivative_dog)))[0]
    maxima_dog = np.array([i for i in zc_dog if derivative_dog[i - 1] > 0 and derivative_dog[i + 1] < 0]).astype(int)
    minima_dog = np.array([i for i in zc_dog if derivative_dog[i - 1] < 0 and derivative_dog[i + 1] > 0]).astype(int)

    fig_dog_combined, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dog_scaled, color='red', label="3×(G₍3.76₎−G₍1.8₎)")
    ax.plot(dog_smooth_scaled, color='green', label="1.8×Smoothed DoG (σₛ=1.5)")
    ax.scatter(maxima_dog, dog_smooth_scaled[maxima_dog], color='cyan', marker='x', s=60, label="Maxima")
    ax.scatter(minima_dog, dog_smooth_scaled[minima_dog], color='blue', marker='x', s=60, label="Minima")
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title("Scaled DoG with Maxima's and Minima's on means:1.98,3.76")
    ax.set_xlabel("Pixel Intensity (0–255)")
    ax.set_ylabel("DoG Value")
    ax.legend()
    centered_plot(fig_dog_combined, "Figure 5B. Scaled DoG with maxima and minima (σ₁=3.76, σ₂=1.8, σₛ=1.5).")

    # ------------- KMeans Fusion Logic -----------------
    def run_kmeans_safe(points, k):
        pts = np.array(points).reshape(-1, 1).astype(float)
        if pts.shape[0] < k:
            full_range = np.linspace(0, 255, 256)
            approx = np.linspace(full_range.min(), full_range.max(), k)
            return np.sort(approx)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pts)
        centers = np.sort(kmeans.cluster_centers_.reshape(-1))
        return centers

    centers_hist = run_kmeans_safe(minima, int(nom_of_objects))
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
        rotated = cv2.warpAffine(text_img, M, (text_w, text_h + baseline), flags=cv2.INTER_LINEAR)
        h, w = rotated.shape[:2]
        if y + h <= img.shape[0] and x + w <= img.shape[1]:
            img[y:y + h, x:x + w] = np.where(rotated > 0, rotated, img[y:y + h, x:x + w])
        return img

    def mean_depth(depthmap, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        return np.mean(depthmap[ly:ry, lx:rx])

    temp = depth_color.copy()
    bounding_boxes = []
    results = []

    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(masks[i])
        x_mm, y_mm = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                          f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        temp = vertical_text(temp, f"Length {int(y_mm)}mm", tl_p)
        cv2.putText(temp, f"Width {int(x_mm)}mm", (tl_p[0], br_p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x_mm), "Length (mm)": int(y_mm)})

    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val = []
    min1 = 255
    for i in range(nom_of_objects):
        _01img = masks[i] // 255
        meanint = depth_color[_01img == 1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref)

    for i in range(nom_of_objects):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm", bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image with width, length, and depth measurements.")

    df = pd.DataFrame(results)
    st.dataframe(df.style.hide(axis='index'), use_container_width=True)

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
