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
st.title("📏 3D Object Measurement (Width, Length, Depth)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))
    img_rgb = initial_image.copy()
    img = Image.fromarray(img_rgb)

    st.subheader("🖼️ Original Image")
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

    # Normalize and colormap
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    st.subheader("🌈 Depth Map (Colorized)")
    st.image(depth_color, use_column_width=True)

    # --- User Inputs ---
    st.sidebar.header("Input Parameters")
    relative_height_ratio = st.sidebar.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    camh = st.sidebar.number_input("Camera Height (mm)", value=300)
    ref_h = st.sidebar.number_input("Reference Object Height (mm)", value=50)
    nom_of_objects = st.sidebar.number_input("Number of Objects", value=1, min_value=1)
    show_steps = st.sidebar.checkbox("Show Intermediate Steps", value=True)

    # --- Histogram & Object Masking ---
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    sigma = 1.89
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    if relative_height_ratio == "low":
        low_bound = 110
        error_rct = 1.08
    elif relative_height_ratio == "med":
        low_bound = 100
        error_rct = 1.23
    elif relative_height_ratio == "high":
        low_bound = 80
        error_rct = 2.91
    else:
        low_bound = 60
        error_rct = 3.0

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound

    # --- Display Histogram and Derivative ---
    if show_steps:
        st.subheader("📊 Histogram & Minima Detection")
        fig, ax = plt.subplots()
        ax.plot(hist, label='Raw Histogram', alpha=0.6)
        ax.plot(smoothed_hist, label='Gaussian Smoothed', linewidth=2)
        ax.scatter(minima, smoothed_hist[minima], color='red', label='Detected Minima')
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 Derivative (DoG-like)")
        fig2, ax2 = plt.subplots()
        ax2.plot(derivative, label='First Derivative')
        ax2.axhline(0, color='k', linestyle='--')
        ax2.legend()
        st.pyplot(fig2)

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1,1))
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))

    # --- Thresholding & Masks ---
    def small_area_remover(binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            output[labels == largest_label] = 255
        return output

    ground_truth = minima[0]
    ret, ground = cv2.threshold(gray, ground_truth, 255, cv2.THRESH_BINARY)

    masks = {}
    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            _, thresh = cv2.threshold(gray, centers[i], 255, cv2.THRESH_BINARY)
            binary = ground - thresh
            masks[i] = small_area_remover(binary)
        sum_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, nom_of_objects):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)

    # --- Show masks if selected ---
    if show_steps:
        st.subheader("⚫ Binary Masks of Objects")
        for i, mask in masks.items():
            st.image(mask, caption=f"Object Mask {i+1}", use_column_width=True)

    # --- Functions ---
    def sad(camheight, depthmap, mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        corners = np.int32(corners)
        x_min, y_min = np.min(corners[:,:,0]), np.min(corners[:,:,1])
        x_max, y_max = np.max(corners[:,:,0]), np.max(corners[:,:,1])
        dx, dy = x_max - x_min, y_max - y_min
        return [dx, dy, (x_min, y_min), (x_max, y_max)]

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144,8.6], cx=0.82, cy=0.79):
        tx, ty = (dx/px)*viewport[1], (dy/py)*viewport[0]
        x = (camh/f)*tx
        y = (camh/f)*ty
        return [cx*x, cy*y]

    # ✅ FIXED VERTICAL TEXT FUNCTION
    def vertical_text(img, text, org, color=(0,255,0)):
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1, 3
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        canvas_h, canvas_w = text_w + text_h, text_w + text_h
        text_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (text_h // 2, canvas_h - baseline - text_h // 2),
                    font, scale, color, thickness, cv2.LINE_AA)
        rotated = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = rotated.shape[:2]
        if y + h > img.shape[0]:
            h = img.shape[0] - y
        if x + w > img.shape[1]:
            w = img.shape[1] - x
        roi = img[y:y+h, x:x+w]
        mask = cv2.cvtColor(rotated[:h, :w], cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(rotated[:h, :w], rotated[:h, :w], mask=mask)
        img[y:y+h, x:x+w] = cv2.add(bg, fg)
        return img

    def mean_depth(depth, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        return np.mean(depth[ly:ry, lx:rx])

    # --- Final Annotation ---
    temp = depth_color.copy()
    bounding_boxes = []
    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(camh, temp, masks[i])
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                    f=5.42, viewport=[6.144,8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0,255,0), 2)
        cv2.putText(temp, f"<Width {int(x)}mm>", (tl_p[0], br_p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        temp = vertical_text(temp, f"<Length {int(y)}mm>", (br_p[0]+10, tl_p[1]))
        bounding_boxes.append([tl_p, br_p])

    ref = mean_depth(depth_color, (0,0), bounding_boxes[0][0])
    mean_val = []
    min1 = 255
    for i in range(nom_of_objects):
        _01img = masks[i]//255
        meanint = depth_color[_01img==1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1-ref)

    for i in range(nom_of_objects):
        temph = (float(mean_val[i]-ref)/scaler)*ref_h
        cv2.putText(temp, f"v Depth {int(temph)}mm v", bounding_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    st.subheader("✅ Final Annotated Image")
    st.image(temp, use_column_width=True)
