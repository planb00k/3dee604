import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

st.title("3D Object Measurement (Width, Length, Depth)")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# --- Take all inputs in one go ---
with st.form("input_form"):
    st.subheader("Enter Input Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
    with col2:
        camh = st.number_input("Camera Height (mm)", value=300)
    with col3:
        ref_h = st.number_input("Reference Object Height (mm)", value=50)
    with col4:
        nom_of_objects = st.number_input("Number of Objects", value=1, min_value=1)
    submitted = st.form_submit_button("Run Measurement")

if uploaded_file and submitted:
    image = Image.open(uploaded_file)
    initial_image = np.array(image.convert("RGB"))
    img_rgb = initial_image.copy()
    img = Image.fromarray(img_rgb)
    st.image(img_rgb, caption="Original Image", use_column_width=True)

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

    # --- Depth visualization ---
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
    st.image(depth_color, caption="Colorized Depth Map", use_column_width=True)

    # --- Histogram ---
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    sigma = 1.89
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    fig, ax = plt.subplots()
    ax.plot(hist, label="Histogram", alpha=0.5)
    ax.plot(smoothed_hist, label="Gaussian Smoothed", color='red')
    ax.legend()
    st.pyplot(fig)

    if relative_height_ratio == "low":
        low_bound = 110
        error_rct = 1.08
    elif relative_height_ratio == "med":
        low_bound = 100
        error_rct = 1.23
    elif relative_height_ratio == "high":
        low_bound = 80
        error_rct = 2.91
    elif relative_height_ratio == "vhigh":
        low_bound = 60

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound

    # --- DoG plot ---
    fig2, ax2 = plt.subplots()
    ax2.plot(derivative, label="DoG (1st Derivative)", color='orange')
    ax2.scatter(minima - low_bound, derivative[minima - low_bound], color='red', label="Detected Minima")
    ax2.legend()
    st.pyplot(fig2)

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1,1))
    labels, centers = kmeans.labels_, kmeans.cluster_centers_
    ascending = np.sort(centers.reshape(len(centers)))

    # --- Mask creation ---
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
            _, thresh = cv2.threshold(gray, ascending[i], 255, cv2.THRESH_BINARY)
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

    for i, mask in masks.items():
        st.image(mask, caption=f"Object Mask {i+1}", use_column_width=True)

    # --- Functions ---
    def sad(camheight, depthmap, mask):
        corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
        corners = np.int32(corners)
        x_min = np.min(corners[:,:,0])
        y_min = np.min(corners[:,:,1])
        x_max = np.max(corners[:,:,0])
        y_max = np.max(corners[:,:,1])
        dx = x_max - x_min
        dy = y_max - y_min
        return [dx, dy, (x_min, y_min), (x_max, y_max)]

    def view(dx, dy, px, py, camh=300, f=6.5, viewport=[6.144,8.6], cx=0.82, cy=0.79):
        tx = (dx/px)*viewport[1]
        ty = (dy/py)*viewport[0]
        x = (camh/f)*tx
        y = (camh/f)*ty
        return [cx*x, cy*y]

    def vertical_text(img, text, org):
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 3
        angle = 90
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (0, text_h), font, scale, (0,255,0), thickness)
        M = cv2.getRotationMatrix2D((text_w//2, text_h//2), angle, 1.0)
        cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
        nW = int((text_h*sin) + (text_w*cos))
        nH = int((text_h*cos) + (text_w*sin))
        M[0,2] += (nW/2) - text_w//2
        M[1,2] += (nH/2) - text_h//2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        h, w = rotated.shape[:2]
        if y+h <= img.shape[0] and x+w <= img.shape[1]:
            img[y:y+h, x:x+w] = np.where(rotated>0, rotated, img[y:y+h, x:x+w])
        return img

    def mean_depth(depth, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        return np.mean(depth[ly:ry, lx:rx])

    # --- Width, Length, Depth ---
    temp = depth_color.copy()
    bounding_boxes = []
    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=masks[i])
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                    f=5.42, viewport=[6.144,8.6], camh=camh)
        cv2.circle(temp, tl_p, 5, (0,255,0), 2)
        cv2.circle(temp, br_p, 5, (0,255,0), 2)
        cv2.rectangle(temp, tl_p, br_p, (0,255,0), 2)
        bounding_boxes.append([tl_p, br_p])
        cv2.putText(temp, f"<Width {int(x)}mm>", (tl_p[0], br_p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        temp = vertical_text(temp, f"<Length {int(y)}mm>", tl_p)

    ref = mean_depth(depth_color, (0,0), bounding_boxes[0][0])
    mean_val = []
    min1 = 255
    for i in range(nom_of_objects):
        _01img = masks[i]//255
        meanint = depth_color[_01img==1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref)

    for i in range(nom_of_objects):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"v Depth {int(temph)}mm v",
                    bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    st.image(temp, caption="Final Annotated Image (with Width, Length, Depth)", use_column_width=True)
