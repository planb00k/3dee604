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
    nom_of_objects = st.number_input("Number of Objects", value=2, min_value=1)
    run_process = st.button("Run Measurement")

# ---------------- Helper Functions ----------------
def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        output[labels == largest_label] = 255
    return output

def find_local_minima(arr):
    g = np.gradient(arr)
    minima_idx = np.where((np.concatenate(([g[0]], g[:-1])) < 0) & (g > 0))[0]
    return minima_idx

def safe_kmeans_centers(points, n_clusters, low=0, high=255):
    if points is None or len(points) == 0:
        return np.linspace(low, high, n_clusters + 1)[1:]
    pts = np.array(points).reshape(-1, 1).astype(float)
    if pts.shape[0] < n_clusters:
        extra_needed = n_clusters - pts.shape[0]
        extra = np.linspace(low, high, extra_needed + 2, dtype=int)[1:-1]
        if extra.size > 0:
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

@st.cache_resource
def load_depth_model():
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    return processor, model

# ---------------- Improved Vertical Label ----------------
def vertical_text(img, text, org, color=(255, 255, 0), angle=90):
    """Draw clear vertical text (rotated phrase with alpha blending)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    pad = 80
    canvas_h, canvas_w = tw + pad * 2, th + pad * 2
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    org_x = pad
    org_y = canvas_h // 2 + th // 2
    cv2.putText(canvas, text, (org_x, org_y), font, scale, (0, 0, 0, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (org_x, org_y), font, scale, (*color, 255), thick, cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((canvas_w / 2, canvas_h / 2), angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    x, y = org
    h, w = rot.shape[:2]
    y = max(0, min(y, img.shape[0] - h))
    x = max(0, min(x, img.shape[1] - w))

    alpha = rot[:, :, 3:] / 255.0
    roi = img[y:y + h, x:x + w]
    roi[:] = (alpha * rot[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    img[y:y + h, x:x + w] = roi
    return img

# ---------------- Run Process ----------------
if run_process and uploaded_file:
    st.info("Processing image. Please wait...")

    image = Image.open(uploaded_file)
    rgb = np.array(image.convert("RGB"))

    processor, model = load_depth_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pp = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth_map = pp[0]["predicted_depth"].squeeze().cpu().numpy()

    d_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9)
    gray = (d_norm * 255).astype(np.uint8)
    color = (plt.cm.magma(d_norm)[:, :, :3] * 255).astype(np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # Histogram & DoG
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s1, s2 = gaussian_filter1d(hist, 3.76), gaussian_filter1d(hist, 1.8)
    dog = 3 * (s1 - s2)
    low_bound = {"low":110,"med":100,"high":80,"vhigh":60}[relative_height_ratio]
    high_bound = 255
    n = int(max(1, nom_of_objects))

    mh = find_local_minima(gaussian_filter1d(hist, 1.89)[low_bound:high_bound])
    mh = (mh + low_bound).astype(int)
    grad = np.gradient(dog)
    zero_cross = np.where(np.diff(np.sign(grad)))[0]
    minima_dog = np.array([i for i in zero_cross if grad[i-1] < 0 and grad[i+1] > 0])
    minima_dog = minima_dog[(minima_dog>=low_bound)&(minima_dog<high_bound)]

    c1 = safe_kmeans_centers(mh, n, low_bound, high_bound)
    c2 = safe_kmeans_centers(minima_dog, n, low_bound, high_bound)
    centers_mid = np.sort((np.array(c1)+np.array(c2))/2).astype(int)

    masks = {}
    ground_th = int(centers_mid[0]) if len(centers_mid)>0 else low_bound
    _, ground = cv2.threshold(gray, ground_th, 255, cv2.THRESH_BINARY)
    if n>1:
        for i in range(1,n):
            thr = int(centers_mid[i]) if i<len(centers_mid) else int(centers_mid[-1])
            _, t = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            b = cv2.subtract(ground, t)
            masks[i] = small_area_remover(b)
        s = np.zeros_like(gray)
        for i in range(1,n):
            s = cv2.add(s, masks[i])
        r = cv2.subtract(ground, s)
        _, r = cv2.threshold(r,1,255,cv2.THRESH_BINARY)
        masks[0] = small_area_remover(r)
    else:
        masks[0]=small_area_remover(ground)
        r=np.zeros_like(gray)

    # Measurement helpers
    def sad(mask):
        if mask is None or np.count_nonzero(mask)==0:
            h,w=gray.shape
            return w,h,(0,0),(w-1,h-1)
        corners=cv2.goodFeaturesToTrack(mask,10,0.05,50)
        if corners is None:
            h,w=gray.shape
            return w,h,(0,0),(w-1,h-1)
        corners=np.int32(corners)
        x_min,y_min=np.min(corners[:,:,0]),np.min(corners[:,:,1])
        x_max,y_max=np.max(corners[:,:,0]),np.max(corners[:,:,1])
        return x_max-x_min,y_max-y_min,(x_min,y_min),(x_max,y_max)

    def view(dx,dy,px,py,camh=300,f=5.42,viewport=[6.144,8.6],cx=0.82,cy=0.79):
        tx,ty=(dx/px)*viewport[1],(dy/py)*viewport[0]
        return [cx*(camh/f)*tx,cy*(camh/f)*ty]

    def mean_depth(dm,lt,rb):
        lx,ly=lt;rx,ry=rb
        lx,rx=max(0,lx),min(dm.shape[1]-1,rx)
        ly,ry=max(0,ly),min(dm.shape[0]-1,ry)
        if ry<=ly or rx<=lx:return float(dm.mean())
        return np.mean(dm[ly:ry,lx:rx])

    temp=color.copy(); boxes=[]; res_tab=[]
    for i in range(n):
        dx,dy,tl,br=sad(masks.get(i,np.zeros_like(gray)))
        x,y=view(dx,dy,rgb.shape[0],rgb.shape[1],camh)
        cv2.rectangle(temp,tl,br,(0,255,0),2)
        boxes.append([tl,br])

        # --- Adaptive vertical placement ---
        cx=(tl[0]+br[0])//2; img_c=rgb.shape[1]//2
        if cx<img_c:
            label_x,angle=br[0]+15,90
        else:
            label_x,angle=tl[0]-60,270

        temp=vertical_text(temp,f"Length {int(y)} mm",(label_x,tl[1]+20),(255,255,0),angle)
        cv2.putText(temp,f"Width {int(x)} mm",(tl[0]+10,br[1]-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        res_tab.append({"Object":i+1,"Width (mm)":int(x),"Length (mm)":int(y)})

    # Depth
    ref=mean_depth(color,(0,0),boxes[0][0])
    means=[]; min1=255
    for i in range(n):
        _m=masks[i]//255
        m=color[_m==1].mean() if np.count_nonzero(_m) else float(color.mean())
        if ref<m<min1:min1=m
        means.append(m)
    scale=float(min1-ref) if (min1-ref)!=0 else 1.0
    for i in range(n):
        d=(float(means[i]-ref)/scale)*ref_h
        cv2.putText(temp,f"Depth {int(d)} mm",boxes[i][0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        res_tab[i]["Depth (mm)"]=int(d)

    # Display
    st.header("Final Annotated Output")
    centered_visual(temp,"Figure 1. Final annotated image showing Width, Length, and Depth values.")

    bbox_only=color.copy()
    for i,(tl,br) in enumerate(boxes):
        cv2.rectangle(bbox_only,tl,br,(0,255,0),2)
        cv2.putText(bbox_only,f"Obj {i+1}",(tl[0],br[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    centered_visual(bbox_only,"Figure 1B. Detected object bounding boxes before annotation.")

    df=pd.DataFrame(res_tab)
    st.dataframe(df.style.hide(axis='index').set_properties(**{'font-size':'16px'}),use_container_width=True)

elif run_process and not uploaded_file:
    st.warning("Please upload an image before running the measurement.")
