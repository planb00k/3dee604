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

# ---------------- Helpers ----------------
def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

def find_local_minima(arr):
    # arr: 1D array
    g = np.gradient(arr)
    minima_idx = np.where((np.concatenate(([g[0]], g[:-1])) < 0) & (g > 0))[0]
    return minima_idx

def safe_kmeans_centers(points, n_clusters, low=0, high=255):
    """
    Return sorted centers (length n_clusters). If not enough points,
    fallback to evenly spaced values between low and high.
    """
    if points is None or len(points) == 0:
        return np.linspace(low, high, n_clusters + 1)[1:]
    pts = np.array(points).reshape(-1, 1).astype(float)
    if pts.shape[0] < n_clusters:
        # augment with evenly spaced values to fill
        extra_needed = n_clusters - pts.shape[0]
        extra = np.linspace(low, high, extra_needed + 2, dtype=int)[1:-1]
        pts = np.vstack([pts, extra.reshape(-1, 1).astype(float)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pts)
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))
    return centers

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
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px; border-radius:6px;">
        <div style="text-align:left; width:{width}px; margin-top:6px;">
            <p style="font-size:16px; font-weight:600;">{caption or ''}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def centered_plot(fig, caption, width=700):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; margin-bottom:40px;">
        <img src="data:image/png;base64,{img_b64}" style="width:{width}px; border-radius:6px;">
        <div style="text-align:left; width:{width}px; margin-top:6px;">
            <p style="font-size:16px; font-weight:600;">{caption or ''}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Run Process ----------------
if run_process and uploaded_file:
    st.info("Processing image. Please wait...")

    # Load image
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

    # Normalize depth and create visuals
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = (plt.cm.magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    # ---------------- Histogram & DoG (as per write-up) ----------------
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # 1) Smoothed histogram (sigma = 1.89)
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    # 2) DoG: (sigma1 = 3.76) - (sigma2 = 1.8)
    sigma_doG_1 = 3.76
    sigma_doG_2 = 1.8
    smoothed_hist1 = gaussian_filter1d(hist, sigma=sigma_doG_1)
    smoothed_hist2 = gaussian_filter1d(hist, sigma=sigma_doG_2)
    dog = smoothed_hist1 - smoothed_hist2

    # 3) Smooth the DoG (sigma = 1.5) and scale for plotting per your report
    smooth_dog = 1.8 * gaussian_filter1d(dog, sigma=1.5)
    scaled_dog = 3 * (smoothed_hist1 - smoothed_hist2)  # for red curve (visual scaling)

    # define low_bound based on relative height ratio (same as earlier logic)
    if relative_height_ratio == "low":
        low_bound = 110
    elif relative_height_ratio == "med":
        low_bound = 100
    elif relative_height_ratio == "high":
        low_bound = 80
    else:
        low_bound = 60
    upper_bound = 255

    # 4) Find minima in both smoothed_hist (windowed) and smooth_dog (windowed)
    mh_window = smoothed_hist[low_bound:upper_bound]
    dog_window = smooth_dog[low_bound:upper_bound]

    minima_hist = find_local_minima(mh_window) + low_bound
    minima_dog = find_local_minima(dog_window) + low_bound

    # guard against empty minima arrays
    minima_hist = np.array(minima_hist).astype(int) if len(minima_hist) > 0 else np.array([], dtype=int)
    minima_dog = np.array(minima_dog).astype(int) if len(minima_dog) > 0 else np.array([], dtype=int)

    # 5) Cluster minima from both methods separately (KMeans) into nom_of_objects centers
    n_clusters = int(max(1, nom_of_objects))
    centers_hist = safe_kmeans_centers(minima_hist, n_clusters, low=low_bound, high=upper_bound)
    centers_dog = safe_kmeans_centers(minima_dog, n_clusters, low=low_bound, high=upper_bound)

    # 6) Take midpoint of corresponding centers to get final thresholds
    # If counts match n_clusters, pair by sorted order
    centers_hist = np.array(centers_hist).astype(float)
    centers_dog = np.array(centers_dog).astype(float)
    # ensure same length
    if centers_hist.shape[0] != n_clusters:
        centers_hist = np.linspace(low_bound, upper_bound, n_clusters + 1)[1:].astype(float)
    if centers_dog.shape[0] != n_clusters:
        centers_dog = np.linspace(low_bound, upper_bound, n_clusters + 1)[1:].astype(float)

    centers_mid = np.sort((centers_hist + centers_dog) / 2.0).astype(int)

    # 7) Use these midpoints as thresholds to create masks (N masks for N objects)
    masks = {}
    # ground threshold - lowest midpoint (foreground)
    ground_threshold = int(centers_mid[0]) if len(centers_mid) > 0 else low_bound
    _, ground = cv2.threshold(gray, ground_threshold, 255, cv2.THRESH_BINARY)

    if n_clusters > 1:
        for i in range(1, n_clusters):
            thr_val = int(centers_mid[i]) if i < len(centers_mid) else int(centers_mid[-1])
            _, thresh = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)

        # residual (unassigned)
        sum_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, n_clusters):
            sum_mask = cv2.add(sum_mask, masks[i])
        residual = cv2.subtract(ground, sum_mask)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks[0] = small_area_remover(ground)
        residual = np.zeros_like(gray)

    # Save hist components for plotting
    hist_components = {
        "hist": hist,
        "smoothed_hist": smoothed_hist,
        "scaled_dog": scaled_dog,
        "smooth_dog": smooth_dog,
        "minima_hist": minima_hist,
        "minima_dog": minima_dog,
        "centers_hist": centers_hist,
        "centers_dog": centers_dog,
        "centers_mid": centers_mid,
        "low_bound": low_bound,
        "upper_bound": upper_bound
    }

    # ---------------- Measurement Functions (preserve old behavior) ----------------
    def sad(camheight, depthmap, mask):
        # goodFeaturesToTrack expects non-zero mask. ensure mask is uint8.
        try:
            if mask is None or np.count_nonzero(mask) == 0:
                # fallback bounding box: full image
                h, w = depthmap.shape[:2]
                return w, h, (0, 0), (w - 1, h - 1)
            corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
            if corners is None:
                h, w = depthmap.shape[:2]
                return w, h, (0, 0), (w - 1, h - 1)
            corners = np.int32(corners)
            x_min = np.min(corners[:, :, 0])
            y_min = np.min(corners[:, :, 1])
            x_max = np.max(corners[:, :, 0])
            y_max = np.max(corners[:, :, 1])
            return x_max - x_min, y_max - y_min, (x_min, y_min), (x_max, y_max)
        except Exception:
            h, w = depthmap.shape[:2]
            return w, h, (0, 0), (w - 1, h - 1)

    # KEEP old view() ordering exactly as your working code (px = shape[0], py = shape[1])
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
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int((text_h * sin) + (text_w * cos))
        nH = int((text_h * cos) + (text_w * sin))
        M[0, 2] += (nW / 2) - text_w // 2
        M[1, 2] += (nH / 2) - text_h // 2
        rotated = cv2.warpAffine(text_img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        h, w = rotated.shape[:2]
        if y + h <= img.shape[0] and x + w <= img.shape[1]:
            img[y:y+h, x:x+w] = np.where(rotated > 0, rotated, img[y:y+h, x:x+w])
        return img

    def mean_depth(depth_map, lt_p, rb_p):
        lx, ly = lt_p
        rx, ry = rb_p
        # clamp indices
        lx = max(0, min(depth_map.shape[1]-1, lx))
        rx = max(0, min(depth_map.shape[1]-1, rx))
        ly = max(0, min(depth_map.shape[0]-1, ly))
        ry = max(0, min(depth_map.shape[0]-1, ry))
        if ry <= ly or rx <= lx:
            return float(depth_map.mean())
        return np.mean(depth_map[ly:ry, lx:rx])

    # ---------------- Measurement and Annotation ----------------
    temp = depth_color.copy()
    bounding_boxes = []
    results = []

    for i in range(n_clusters):
        mask_i = masks.get(i, np.zeros_like(gray))
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=mask_i)
        # preserve old px/py ordering
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1],
                    f=5.42, viewport=[6.144, 8.6], camh=camh)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        results.append({"Object": i + 1, "Width (mm)": int(x), "Length (mm)": int(y)})
        temp = vertical_text(temp, f"Length {int(y)}mm", tl_p)
        cv2.putText(temp, f"Width {int(x)}mm", (tl_p[0], br_p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Depth calculation (preserve original method)
    ref = mean_depth(depth_color, (0, 0), bounding_boxes[0][0])
    mean_val = []
    min1 = 255
    for i in range(n_clusters):
        _01img = masks[i] // 255
        if np.count_nonzero(_01img) == 0:
            meanint = float(depth_color.mean())
        else:
            meanint = depth_color[_01img == 1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_val.append(meanint)
    scaler = float(min1 - ref) if (min1 - ref) != 0 else 1.0

    for i in range(n_clusters):
        temph = (float(mean_val[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"Depth {int(temph)}mm",
                    bounding_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        results[i]["Depth (mm)"] = int(temph)

    # ---------------- Display Section (restore all intermediate visuals) ----------------
    st.header("Final Annotated Output")
    centered_visual(temp, "Figure 1. Final annotated image showing calculated Width, Length, and Depth values for detected objects.")

    # Bounding boxes only
    bbox_only = depth_color.copy()
    for i, (tl, br) in enumerate(bounding_boxes):
        cv2.rectangle(bbox_only, tl, br, (0, 255, 0), 2)
        cv2.putText(bbox_only, f"Obj {i+1}", (tl[0], br[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    centered_visual(bbox_only, "Figure 1B. Detected object bounding boxes before dimension annotation.")

    df = pd.DataFrame(results)
    st.markdown("<h5 style='font-size:20px;'>Object Dimension Measurements</h5>", unsafe_allow_html=True)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size': '16px'}), use_container_width=True)

    st.markdown("---")
    st.header("Intermediate Visualizations")

    with st.expander("Original and Depth Representations", expanded=False):
        centered_visual(initial_image, "Figure 2. Original RGB image used for depth analysis.")
        centered_visual(depth_gray, "Figure 3. Grayscale depth map representing normalized pixel depth values.")
        centered_visual(depth_color, "Figure 4. Colorized depth map using magma colormap for visualizing relative distances.")

    with st.expander("Depth Intensity Histogram & Smoothed", expanded=False):
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.plot(hist, label="Raw Histogram", alpha=0.6)
        ax_hist.plot(smoothed_hist, label="Gaussian Smoothed (σ=1.89)", color='red', linewidth=2)
        ax_hist.set_title("Depth Intensity Distribution")
        ax_hist.set_xlabel("Pixel Intensity (0–255)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        centered_plot(fig_hist, "Figure 5. Raw and smoothed histogram showing intensity distribution of the grayscale depth map.")

    with st.expander("DoG (Report-style) Visualization", expanded=False):
        fig_dog, ax_dog = plt.subplots(figsize=(10, 4))
        ax_dog.plot(scaled_dog, color='red', label='3×(Gσ1 - Gσ2)')  # red scaled
        ax_dog.plot(smooth_dog, color='green', label='1.8×Smoothed DoG (σ=1.5)')  # smoothed dog
        # minima and maxima markers on smooth_dog (offset safe)
        if len(hist_components["minima_dog"]) if "minima_dog" in locals() else len(minima_dog):
            md = minima_dog if len(minima_dog) > 0 else np.array([])
            ax_dog.plot(md, smooth_dog[md], 'bx', label='Minima (DoG)', markersize=6)
        if len(minima_hist) > 0:
            ax_dog.plot(minima_hist, smoothed_hist[minima_hist], 'cx', label='Minima (Smoothed Hist)', markersize=6)
        # cluster centers and midpoints
        ax_dog.set_title("Scaled DoG with Maxima and Minima (σ1=3.76, σ2=1.8) and midpoints")
        ax_dog.set_xlabel("Intensity bins (0–255)")
        ax_dog.set_ylabel("Amplitude")
        ax_dog.legend()
        centered_plot(fig_dog, "Figure 6. DoG + Smoothed-DoG showing minima used in segmentation (per write-up).")

    with st.expander("Segmentation and Object Masks", expanded=False):
        centered_visual(ground, "Figure 7. Ground threshold mask after initial binary segmentation.")
        for key, mask in sorted(masks.items(), key=lambda x: x[0]):
            centered_visual(mask, f"Figure 8.{key + 1} Object Mask {key + 1} after area refinement using connected components.")
        centered_visual(residual, "Figure 9. Residual mask showing unassigned or background regions after segmentation.")

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
