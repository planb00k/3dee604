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

    # ---------------- Histogram & DoG (Corrected) ----------------
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    example_hist = hist.copy()

    sigma1, sigma2 = 1.98, 3.76
    smoothed_hist1 = gaussian_filter1d(example_hist, sigma=sigma1)
    smoothed_hist2 = gaussian_filter1d(example_hist, sigma=sigma2)
    dog = smoothed_hist1 - smoothed_hist2
    scaled_dog = 3 * dog
    smooth_dog = 1.8 * gaussian_filter1d(dog, sigma=1.5)

    low_bound = 60 if relative_height_ratio == "vhigh" else (
        80 if relative_height_ratio == "high" else (
        100 if relative_height_ratio == "med" else 110))
    upper_bound = 255

    derivative1 = np.gradient(smooth_dog[low_bound:upper_bound])
    zero_crossings1 = np.where(np.diff(np.sign(derivative1)))[0]
    maxima1 = np.array([i for i in zero_crossings1 if derivative1[i-1] > 0 and derivative1[i+1] < 0]).astype(int) + low_bound
    minima1 = np.array([i for i in zero_crossings1 if derivative1[i-1] < 0 and derivative1[i+1] > 0]).astype(int) + low_bound

    if len(minima1) >= nom_of_objects:
        kmeans1 = KMeans(n_clusters=nom_of_objects, random_state=42)
        kmeans1.fit(minima1.reshape(-1, 1))
        centers1 = np.sort(kmeans1.cluster_centers_.reshape(-1))
    else:
        centers1 = np.linspace(low_bound, upper_bound, num=nom_of_objects + 1, dtype=int)[1:]

    def small_area_remover(binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            output[labels == largest_label] = 255
        return output

    ground_truth = int(minima1[0]) if len(minima1) > 0 else int(low_bound)
    _, ground = cv2.threshold(gray, ground_truth, 255, cv2.THRESH_BINARY)
    masks = {}

    if nom_of_objects > 1:
        for i in range(1, len(centers1)):
            _, thresh = cv2.threshold(gray, int(centers1[i]), 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)

        sum_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, len(centers1)):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)
        residual = np.zeros_like(gray)

    hist_components = {
        "hist": hist,
        "scaled_dog": scaled_dog,
        "smooth_dog": smooth_dog,
        "maxima1": maxima1,
        "minima1": minima1
    }

    # ---------------- Measurement Functions (Old logic kept) ----------------
    def sad(camheight, depthmap, mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        corners = np.int32(corners)
        x_min = np.min(corners[:, :, 0])
        y_min = np.min(corners[:, :, 1])
        x_max = np.max(corners[:, :, 0])
        y_max = np.max(corners[:, :, 1])
        return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)

    def view(dx, dy, px, py, camh=300, f=5.42, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
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
        rotated = cv2.warpAffine(text_img, M, (text_w, text_h), flags=cv2.INTER_LINEAR)
        h, w = rotated.shape[:2]
        img[y:y+h, x:x+w] = np.where(rotated > 0, rotated, img[y:y+h, x:x+w])
        return img

    def mean_depth(depth, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        return np.mean(depth[ly:ry, lx:rx])

    # ---------------- Measurement and Annotation ----------------
    temp = depth_color.copy()
    bounding_boxes, results = [], []

    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=masks[i])
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                    f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x), "Length (mm)": int(y)})
        temp = vertical_text(temp, f"Length {int(y)}mm", tl_p)
        cv2.putText(temp, f"Width {int(x)}mm", (tl_p[0], br_p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val, min1 = [], 255
    for i in range(nom_of_objects):
        _01img = masks[i] // 255
        meanint = depth_color[_01img == 1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0

    for i in range(nom_of_objects):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm",
                    bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

    # ---------------- Display Section ----------------
    def centered_visual(img_array, caption=None, width=550):
        if isinstance(img_array, np.ndarray):
            img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        else:
            img_pil = img_array
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
            f"<div style='text-align:center; margin-bottom:40px;'>"
            f"<img src='data:image/png;base64,{img_b64}' style='width:{width}px; border-radius:6px;'>"
            f"<p style='font-weight:bold; font-size:18px'>{caption}</p></div>",
            unsafe_allow_html=True)

    def centered_plot(fig, caption, width=550):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        st.markdown(
            f"<div style='text-align:center; margin-bottom:40px;'>"
            f"<img src='data:image/png;base64,{img_b64}' style='width:{width}px; border-radius:6px;'>"
            f"<p style='font-weight:bold; font-size:18px'>{caption}</p></div>",
            unsafe_allow_html=True)

    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values.")

    df = pd.DataFrame(results)
    st.markdown("<h5 style='font-size:20px;'>Object Dimension Measurements</h5>", unsafe_allow_html=True)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '16px'}), use_container_width=True)

    with st.expander("DoG Visualization", expanded=False):
        fig_dog, ax_dog = plt.subplots(figsize=(8, 4))
        ax_dog.plot(hist_components["scaled_dog"], color='red', label='3×(σ₁–σ₂) DoG')
        ax_dog.plot(hist_components["smooth_dog"], color='green', label='1.8×Smoothed DoG (σ=1.5)')
        ax_dog.plot(hist_components["maxima1"], 1.8*hist_components["smooth_dog"][hist_components["maxima1"]], 'cx', label='Maxima', markersize=6)
        ax_dog.plot(hist_components["minima1"], 1.8*hist_components["smooth_dog"][hist_components["minima1"]], 'bx', label='Minima', markersize=6)
        ax_dog.set_title("Scaled DoG with Maxima and Minima (σ₁=1.98, σ₂=3.76, σ_smooth=1.5)")
        ax_dog.legend()
        centered_plot(fig_dog, "Figure 6. Scaled DoG plot identical to report illustration.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
