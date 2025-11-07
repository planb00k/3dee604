# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("3D Object Measurement — Width, Length, Depth (Notebook-matching)")

# ---------------- Input form (all inputs in one go) ----------------
with st.form("params"):
    uploaded_file = st.file_uploader("Upload image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])
    c1, c2, c3 = st.columns(3)
    with c1:
        relative_height_ratio = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"], index=1)
        nom_of_objects = st.number_input("Number of objects", min_value=1, max_value=10, value=1)
    with c2:
        camh = st.number_input("Camera height (camh) [mm]", min_value=1, value=289)
        ref_h = st.number_input("Reference object height ref_h [mm]", min_value=1.0, value=100.0)
    with c3:
        show_intermediate = st.checkbox("Show intermediate debug visuals (optional)", value=False)
        run = st.form_submit_button("Run")

if not run:
    st.info("Upload an image, configure parameters and click Run.")
    st.stop()

if uploaded_file is None:
    st.error("Please upload an image to run the pipeline.")
    st.stop()

# ---------------- Read image ----------------
image = Image.open(uploaded_file).convert("RGB")
initial_image = np.array(image)  # RGB (H,W,3)
h_img, w_img = initial_image.shape[0], initial_image.shape[1]

# ---------------- Load model & estimate depth ----------------
# If this fails because of environment/model download issues, switch to a cached/local model or Intel/dpt-small
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
depth_result = post_processed[0]
if "predicted_depth" in depth_result:
    depth_raw = depth_result["predicted_depth"].squeeze().cpu().numpy()
elif "depth" in depth_result:
    depth_raw = depth_result["depth"].squeeze().cpu().numpy()
else:
    st.error(f"Depth key missing. Keys: {depth_result.keys()}")
    st.stop()

# ---------------- Create index map (0..255) consistent with colormap and visualization ----------------
# Using normalized indices is how the notebook computed means from depth_color grayscale.
depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-12)
depth_index = (depth_norm * 255.0).astype(np.float32)  # keep float for mean calculations
# small median blur on index map to reduce isolated noisy pixels
depth_index_blur = cv2.medianBlur(depth_index.astype(np.uint8), 3).astype(np.float32)

# colorized depth (magma) for display (RGB)
depth_magma = plt.cm.magma(depth_norm)
depth_color_rgb = (depth_magma[:, :, :3] * 255).astype(np.uint8)
depth_color_bgr = cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR)

# ---------------- Histogram & smoothed minima detection ----------------
gray = (depth_index_blur).astype(np.uint8)  # use blurred index map for histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
sigma = 1.89
smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

if relative_height_ratio == "low":
    low_bound = 110; error_rct = 1.08
elif relative_height_ratio == "med":
    low_bound = 100; error_rct = 1.23
elif relative_height_ratio == "high":
    low_bound = 80; error_rct = 2.91
else:
    low_bound = 60; error_rct = 1.0

derivative = np.gradient(smoothed_hist[low_bound:])
zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
if len(zero_crossings) >= 3:
    maxima = np.array([i for i in zero_crossings if derivative[i-1] > 0 and derivative[i+1] < 0]).astype(int) + low_bound
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound
else:
    maxima = np.array([])
    minima = np.array([low_bound])

# If minima are fewer than objects, fallback for KMeans
if len(minima) < max(1, nom_of_objects):
    minima_for_kmeans = np.linspace(low_bound, 255, num=max(1, nom_of_objects)).astype(int)
else:
    minima_for_kmeans = minima

# KMeans (rounded to int thresholds)
try:
    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima_for_kmeans.reshape(-1,1))
    centers = np.sort(np.round(kmeans.cluster_centers_.reshape(-1)).astype(int))
except Exception:
    centers = np.linspace(low_bound, 255, num=nom_of_objects).astype(int)

# DoG for plotting explanation
sigma1 = 1.8; sigma2 = 3.0
smoothed_hist1 = gaussian_filter1d(hist, sigma=sigma1)
smoothed_hist2 = gaussian_filter1d(hist, sigma=sigma2)
dog = smoothed_hist1 - smoothed_hist2

# ---------------- Thresholding / Mask creation ----------------
ground_truth = int(minima[0]) if len(minima) > 0 else int(low_bound)
_, ground = cv2.threshold(gray, ground_truth, 255, cv2.THRESH_BINARY)

def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

masks = {}
if nom_of_objects > 1:
    for i in range(1, nom_of_objects):
        thresh_val = int(centers[i]) if i < len(centers) else int(centers[-1])
        _, thr = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        binary = cv2.subtract(ground, thr)
        masks[i] = small_area_remover(binary)
    sum_img = np.zeros(gray.shape, dtype=np.uint8)
    for i in range(1, nom_of_objects):
        sum_img = cv2.add(sum_img, masks[i])
    residual = cv2.subtract(ground, sum_img)
    _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
    masks[0] = small_area_remover(residual)
else:
    masks[0] = small_area_remover(ground)

mask_list_for_display = [masks[i] for i in range(nom_of_objects)]

# ---------------- Helper functions (preserve notebook logic) ----------------
def merge_colinear_lines(lines, angle_threshold=5, distance_threshold=20):
    if lines is None:
        return []
    merged_lines = []
    def line_angle(l):
        x1, y1, x2, y2 = l
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))
    def endpoint_distance(l1, l2):
        x11, y11, x12, y12 = l1
        x21, y21, x22, y22 = l2
        dist1 = np.hypot(x11 - x21, y11 - y21)
        dist2 = np.hypot(x11 - x22, y11 - y22)
        dist3 = np.hypot(x12 - x21, y12 - y21)
        dist4 = np.hypot(x12 - x22, y12 - y22)
        dists = np.array([dist1,dist2,dist3,dist4])
        return np.min(dists)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for i, mline in enumerate(merged_lines):
            if abs(line_angle(line[0]) - line_angle(mline)) < angle_threshold and endpoint_distance(line[0], mline) < distance_threshold:
                all_points = np.array([[x1, y1], [x2, y2], [mline[0], mline[1]], [mline[2], mline[3]]])
                x_coords = all_points[:,0]; y_coords = all_points[:,1]
                if abs(x_coords[0] - x_coords[1]) > abs(y_coords[0] - y_coords[1]):
                    idx_min, idx_max = np.argmin(x_coords), np.argmax(x_coords)
                else:
                    idx_min, idx_max = np.argmin(y_coords), np.argmax(y_coords)
                merged_lines[i] = [x_coords[idx_min], y_coords[idx_min], x_coords[idx_max], y_coords[idx_max]]
                merged = True
                break
        if not merged:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines

def sad(camheight, depthmap, mask, viewport=[3.4,3.6], f=6.5, imgsize=[h_img, w_img]):
    bing = mask
    corners = None
    try:
        corners = cv2.goodFeaturesToTrack(bing, 10, 0.05, 50)
    except:
        corners = None
    if corners is None:
        ys, xs = np.where(bing > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0,0,(0,0),(imgsize[1]-1, imgsize[0]-1)]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return [x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)]
    corners = np.int32(corners)
    binary = cv2.cvtColor(bing, cv2.COLOR_GRAY2BGR)
    imgl = binary.copy()
    edges_local = cv2.Canny(binary, 20, 50, apertureSize=3)
    linesP = cv2.HoughLinesP(edges_local, 0.5, np.pi / 720, threshold=10, minLineLength=70, maxLineGap=20)
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv2.line(imgl, (x1, y1), (x2, y2), (0,255,0), 2)
    imglm = imgl.copy()
    if linesP is not None:
        merged_lines = merge_colinear_lines(linesP, 15, 400)
        for x1, y1, x2, y2 in merged_lines:
            cv2.line(imglm, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    x_min = np.min(corners[:,:,0]); y_min = np.min(corners[:,:,1])
    x_max = np.max(corners[:,:,0]); y_max = np.max(corners[:,:,1])
    return [int(x_max-x_min), int(y_max-y_min), (int(x_min), int(y_min)), (int(x_max), int(y_max))]

def view(dx, dy, px, py, camh=300, cx=0.82, cy=0.79, f=6.5, viewport=[3.6,6.4]):
    tx = (dx/px) * viewport[1]
    ty = (dy/py) * viewport[0]
    x = (camh/f) * tx
    y = (camh/f) * ty
    return [(cx)*x, (cy)*y]

def vertical_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0,255,0),
                  thickness=3, lineType=cv2.LINE_AA, method="rotate", spacing=40, angle=90):
    x, y = org
    img_out = img.copy()
    if method == "stack":
        for ch in text:
            cv2.putText(img_out, ch, (x, y), font, scale, color, thickness, lineType)
            y += spacing
    elif method == "rotate":
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (0, text_h), font, scale, color, thickness, lineType)
        M = cv2.getRotationMatrix2D((text_w//2, text_h//2), angle, 1.0)
        cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
        nW = int((text_h * sin) + (text_w * cos))
        nH = int((text_h * cos) + (text_w * sin))
        M[0,2] += (nW / 2) - text_w // 2
        M[1,2] += (nH / 2) - text_h // 2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
        h, w = rotated.shape[:2]
        x2 = min(max(0, x), img_out.shape[1] - w - 1)
        y2 = min(max(0, y), img_out.shape[0] - h - 1)
        mask_rot = rotated[:,:,0] > 0
        for c in range(3):
            img_out[y2:y2+h, x2:x2+w, c] = np.where(mask_rot, rotated[:,:,c], img_out[y2:y2+h, x2:x2+w, c])
    else:
        raise ValueError("method must be 'rotate' or 'stack'")
    return img_out

def mean_depth(depth_arr, lt_p, rb_p):
    lx, ly = lt_p; rx, ry = rb_p
    lx, rx = max(0,lx), min(depth_arr.shape[1]-1, rx)
    ly, ry = max(0,ly), min(depth_arr.shape[0]-1, ry)
    if lx >= rx or ly >= ry:
        return float(depth_arr.mean())
    return float(np.mean(depth_arr[ly:ry, lx:rx]))

# ---------------- Final annotated image (match notebook order) ----------------
bounding_boxes = []
temp = depth_color_bgr.copy()

for i in range(nom_of_objects):
    dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=masks[i])
    # NOTE: to match original notebook we pass px=h_img, py=w_img (that's how notebook used it)
    x_mm, y_mm = view(dx, dy, px=h_img, py=w_img, f=5.42, viewport=[6.144,8.6], camh=camh)
    cv2.circle(temp, tl_p, 5, (0,255,0), 2)
    cv2.circle(temp, br_p, 5, (0,255,0), 2)
    cv2.rectangle(temp, tl_p, br_p, (0,255,0), 2)
    bounding_boxes.append([tl_p, br_p])
    # Width text (horizontal)
    cv2.putText(temp, f"<Width {int(x_mm)}mm>", (tl_p[0], br_p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    # Length text (vertical) - restored
    temp = vertical_text(temp, f"<Length {int(y_mm)}>mm", tl_p)

# ---------------- Depth calculation using index map (0..255) ----------------
# Reference from the first bounding box top-left area (preserve original notebook intent)
ref = mean_depth(depth_index_blur, (0,0), bounding_boxes[0][0])  # using index map for consistency
mean_val = []
min1 = 255.0
for i in range(nom_of_objects):
    _01img = (masks[i] // 255).astype(bool)
    if np.sum(_01img) == 0:
        meanint = float(depth_index_blur.mean())
    else:
        meanint = float(depth_index_blur[_01img].mean())
    if ref < meanint < min1:
        min1 = meanint
    mean_val.append(meanint)

scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0

for i in range(nom_of_objects):
    temph = (float(mean_val[i] - ref) / scaler) * ref_h
    # Depth text (placed at top-left of bbox)
    cv2.putText(temp, f"v Depth {int(temph)}mm v", bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

# ---------------- Analysis plots ----------------
# Histogram figure
fig_hist, ax_hist = plt.subplots(figsize=(10,4))
ax_hist.plot(hist, label='Original Histogram', alpha=0.5)
ax_hist.plot(smoothed_hist, label=f'Gaussian Blurred Histogram (σ={sigma})', linewidth=2)
if len(minima) > 0:
    ax_hist.plot(minima, smoothed_hist[minima], 'rx', label='Minima', markersize=8)
ax_hist.set_title('Grayscale Image Histogram and Gaussian-Smoothed Version')
ax_hist.legend()
plt.tight_layout()

# DoG figure
fig_dog, ax_dog = plt.subplots(figsize=(10,4))
ax_dog.plot(smoothed_hist1, label=f'sigma={sigma1}')
ax_dog.plot(smoothed_hist2, label=f'sigma={sigma2}')
ax_dog.plot(dog, color='red', label='DoG')
ax_dog.set_title('DoG (Difference of Gaussians)')
ax_dog.legend()
plt.tight_layout()

# Located indices bar
fig_bar, ax_bar = plt.subplots(figsize=(8,1.5))
values = np.linspace(0, 255, 256).reshape(1, -1)
ax_bar.imshow(values, cmap='magma', aspect='auto')
for mv in mean_val:
    ax_bar.axvline(x=mv, color='lime', linewidth=3)
ax_bar.axvline(x=ref, color='cyan', linewidth=3)
ax_bar.set_yticks([]); ax_bar.set_xticks([0,64,128,192,255])
ax_bar.set_xticklabels(['0','64','128','192','255'])
ax_bar.set_title('Located Indices')
plt.tight_layout()

# ---------------- Layout: Main visuals + Analysis + Logs ----------------
st.markdown("## Main visuals")
colA, colB = st.columns(2)
with colA:
    st.subheader("Colorized depth map (magma)")
    st.image(depth_color_rgb, use_column_width=True)
    st.subheader("Thresholded binary segmentation")
    st.image(cv2.cvtColor(ground, cv2.COLOR_GRAY2RGB), use_column_width=True)
with colB:
    st.subheader("Individual object masks")
    mask_cols = st.columns(min(4, max(1, nom_of_objects)))
    for idx, mask_img in enumerate(mask_list_for_display):
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        col = mask_cols[idx % len(mask_cols)]
        col.image(mask_rgb, caption=f"Mask {idx+1}", use_column_width=True)

st.markdown("---")
st.subheader("Final annotated image (Width / Length / Depth)")
st.image(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB), use_column_width=True)

st.markdown("---")
st.subheader("Analysis plots")
ap_cols = st.columns(3)
with ap_cols[0]:
    st.pyplot(fig_hist)
with ap_cols[1]:
    st.pyplot(fig_dog)
with ap_cols[2]:
    st.pyplot(fig_bar)

with st.expander("Logs & intermediate values (minima, centers, scaler, raw arrays)", expanded=False):
    st.write("minima:", minima.tolist() if isinstance(minima, np.ndarray) else minima)
    st.write("kmeans centers (rounded):", centers.tolist())
    st.write("ground_truth (threshold index):", int(ground_truth))
    st.write("ref (reference index mean):", float(ref))
    st.write("mean_val per object (index means):", mean_val)
    st.write("scaler:", scaler)

# Download final annotated image
buf = cv2.imencode('.png', cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))[1].tobytes()
st.download_button("Download final annotated image", data=buf, file_name="annotated_result.png", mime="image/png")
