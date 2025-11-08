import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("3D Object Measurement (Width, Length, Depth)")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Single-submit form so all inputs are taken in one go
with st.form("inputs"):
    st.subheader("Input parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    with col2:
        camh = st.number_input("Camera Height (mm)", value=300)
    with col3:
        ref_h = st.number_input("Reference Object Height (mm)", value=50)
    with col4:
        nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    show_steps = st.checkbox("Show intermediate steps", value=True)
    submitted = st.form_submit_button("Run")

if uploaded_file and submitted:
    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))
    h_img, w_img = initial_image.shape[0], initial_image.shape[1]
    img = Image.fromarray(initial_image)

    st.subheader("Original image")
    st.image(initial_image, use_column_width=True)

    # --- Depth Estimation ---
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)]
    )

    depth_result = post_processed[0]
    if "predicted_depth" in depth_result:
        depth = depth_result["predicted_depth"].squeeze().cpu().numpy()
    elif "depth" in depth_result:
        depth = depth_result["depth"].squeeze().cpu().numpy()
    else:
        raise KeyError(f"Depth key missing: {depth_result.keys()}")

    # normalize depth (float), but keep raw float for numeric ops
    depth_min, depth_max = float(depth.min()), float(depth.max())
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)  # 0..1 floats

    # For display & histogram use 0..255 uint8 representation of normalized depth
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color_bgr = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    st.subheader("Colorized depth map")
    st.image(depth_color_bgr, use_column_width=True)

    # --- Histogram & smoothing on numeric depth (uint8) ---
    gray = depth_uint8.copy()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    sigma = 1.89
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    if show_steps:
        fig, ax = plt.subplots()
        ax.plot(hist, label="Histogram (depth uint8)", alpha=0.6)
        ax.plot(smoothed_hist, label="Smoothed", linewidth=2)
        ax.legend()
        st.pyplot(fig)

    # low bound selection
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
    # safe minima detection (check bounds)
    minima_candidates = []
    for idx in zero_crossings:
        if idx - 1 >= 0 and idx + 1 < len(derivative):
            if derivative[idx-1] < 0 and derivative[idx+1] > 0:
                minima_candidates.append(idx + low_bound)
    minima = np.array(minima_candidates, dtype=int)

    if show_steps:
        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(derivative)) + low_bound, derivative, label="Derivative")
        ax2.scatter(minima, derivative[minima - low_bound], color="red", label="Minima")
        ax2.legend()
        st.pyplot(fig2)

    # if minima too small, fallback to simple threshold tiers
    if len(minima) == 0:
        # simple fallback thresholds
        minima = np.array([low_bound, min(180, low_bound+30)], dtype=int)

    # KMeans clustering on minima positions -> choose cluster centers as thresholds
    # ensure we have enough minima to cluster (KMeans requires >= n clusters)
    min_vals_for_kmeans = minima.reshape(-1, 1)
    if len(min_vals_for_kmeans) < nom_of_objects:
        # pad with evenly spaced values between low_bound and 220
        extra = max(0, nom_of_objects - len(min_vals_for_kmeans))
        pad = np.linspace(low_bound, 220, extra + len(min_vals_for_kmeans) + 2)[1:-1]
        min_vals_for_kmeans = pad.reshape(-1,1)

    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(min_vals_for_kmeans)
    centers = kmeans.cluster_centers_.reshape(-1)
    centers_int = np.sort(np.round(centers).astype(int))

    # Create masks using thresholds on depth_uint8 (NOT color image)
    def small_area_remover(binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            output[labels == largest_label] = 255
        return output

    ground_truth = minima[0] if len(minima) > 0 else centers_int[0]
    _, ground = cv2.threshold(gray, int(ground_truth), 255, cv2.THRESH_BINARY)
    masks = {}

    if nom_of_objects > 1:
        # Use centers_int as thresholds for layered segmentation
        for i in range(1, nom_of_objects):
            thr = int(centers_int[i]) if i < len(centers_int) else int(centers_int[-1])
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

    # show masks
    if show_steps:
        st.subheader("Object masks (binary)")
        for i, mask in masks.items():
            st.image(mask, caption=f"Mask {i+1}", use_column_width=True)

    # --- helper functions ---
    def safe_bbox_from_mask(mask):
        # if goodFeatures fails, fallback to contours bbox
        corners = cv2.goodFeaturesToTrack(mask, 50, 0.01, 10)
        if corners is None:
            # use contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return (x, y), (x + w, y + h)
        corners = np.int32(corners)
        x_min = int(np.min(corners[:, :, 0]))
        y_min = int(np.min(corners[:, :, 1]))
        x_max = int(np.max(corners[:, :, 0]))
        y_max = int(np.max(corners[:, :, 1]))
        return (x_min, y_min), (x_max, y_max)

    def sad(camheight, depthmap, mask):
        bbox = safe_bbox_from_mask(mask)
        if bbox is None:
            return [0, 0, (0, 0), (0, 0)]
        (x_min, y_min), (x_max, y_max) = bbox
        dx = x_max - x_min
        dy = y_max - y_min
        return [dx, dy, (x_min, y_min), (x_max, y_max)]

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144, 8.6], cx=0.82, cy=0.79):
        # px = image width, py = image height
        if px == 0 or py == 0:
            return [0.0, 0.0]
        tx = (dx / px) * viewport[1]
        ty = (dy / py) * viewport[0]
        x = (camh / f) * tx
        y = (camh / f) * ty
        return [cx * x, cy * y]

    def vertical_text(img, text, org, color=(0, 255, 0)):
        # Robust vertical text drawing (90deg CCW) with clipping
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.8, 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        # create canvas and write text then rotate
        canvas_h, canvas_w = text_h + 6, text_w + 6
        text_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (3, canvas_h - 3), font, scale, color, thickness, cv2.LINE_AA)
        rotated = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = rotated.shape[:2]
        # clip to image bounds
        y_end = min(y + h, img.shape[0])
        x_end = min(x + w, img.shape[1])
        y_start = max(y, 0)
        x_start = max(x, 0)
        if y_start >= y_end or x_start >= x_end:
            return img
        roi = img[y_start:y_end, x_start:x_end]
        rot_crop = rotated[0:(y_end - y_start), 0:(x_end - x_start)]
        mask = cv2.cvtColor(rot_crop, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_bin)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(rot_crop, rot_crop, mask=mask_bin)
        img[y_start:y_end, x_start:x_end] = cv2.add(bg, fg)
        return img

    def mean_depth(depth_arr, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        # clip coordinates
        lx = max(0, min(int(lx), depth_arr.shape[1]-1))
        rx = max(0, min(int(rx), depth_arr.shape[1]-1))
        ly = max(0, min(int(ly), depth_arr.shape[0]-1))
        ry = max(0, min(int(ry), depth_arr.shape[0]-1))
        if rx <= lx or ry <= ly:
            return float(np.nan)
        return float(np.mean(depth_arr[ly:ry, lx:rx]))

    # --- Final annotations (use numeric depth values for mean depth) ---
    temp = depth_color_bgr.copy()
    bounding_boxes = []
    for i in range(nom_of_objects):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=depth_uint8, mask=mask_i)
        # pass px=width, py=height (correct)
        x_mm, y_mm = view(dx, dy, px=w_img, py=h_img, camh=camh)
        # draw bbox and points
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        cv2.circle(temp, tl_p, 4, (0, 0, 255), -1)
        cv2.circle(temp, br_p, 4, (255, 0, 0), -1)
        bounding_boxes.append([tl_p, br_p])
        # write width near bottom-left
        bottom_left = (tl_p[0], min(temp.shape[0]-5, br_p[1]+20))
        cv2.putText(temp, f"Width {int(round(x_mm))} mm", bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        # write length vertically at left edge inside bbox (try inside first)
        mid_x = int((tl_p[0] + br_p[0]) / 2)
        mid_y = int((tl_p[1] + br_p[1]) / 2)
        # try vertical text inside box; fallback to right side if not fit
        temp = vertical_text(temp, f"Length {int(round(y_mm))} mm", (tl_p[0]+5, tl_p[1]+5))

    # compute depth values from original float depth (not color)
    # reference depth = mean depth of top-left area of first bounding box
    if len(bounding_boxes) == 0:
        st.error("No objects detected — adjust number of objects or parameters.")
    else:
        ref_mean = mean_depth(depth, (0, 0), bounding_boxes[0][0])
        mean_vals = []
        min1 = 1e9
        for i in range(nom_of_objects):
            mask_i = masks.get(i, np.zeros_like(gray))
            mask_bool = mask_i.astype(bool)
            if mask_bool.sum() == 0:
                meanint = np.nan
            else:
                # mean of original float depth — better numeric semantics
                meanint = float(np.mean(depth[mask_bool]))
            mean_vals.append(meanint)
            if not np.isnan(meanint) and (ref_mean is not None) and (ref_mean < meanint < min1):
                min1 = meanint
        if np.isfinite(min1) and (not np.isnan(ref_mean)):
            scaler = float(min1 - ref_mean) if (min1 - ref_mean) != 0 else 1.0
        else:
            scaler = 1.0

        for i in range(nom_of_objects):
            if np.isnan(mean_vals[i]) or np.isnan(ref_mean):
                temph_val = 0
            else:
                temph_val = (float(mean_vals[i] - ref_mean) / scaler) * ref_h
            cv2.putText(temp, f"Depth {int(round(temph_val))} mm", (bounding_boxes[i][0][0], bounding_boxes[i][0][1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

        st.subheader("Final annotated image")
        st.image(temp, use_column_width=True)
