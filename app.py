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
import io, base64

st.set_page_config(page_title="3D Object Measurement", layout="wide")
st.title("3D Object Measurement (Width, Length, Depth)")

# ----------------- helper display -----------------
def show_img(img, cap="", w=650):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"<div style='text-align:center'><img src='data:image/png;base64,{data}' width={w}><br><b>{cap}</b></div>", unsafe_allow_html=True)

# ----------------- model cache -----------------
@st.cache_resource
def load_model():
    proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return proc, model, device

# ----------------- functions -----------------
def kmeans_centers(points, k):
    pts = np.array(points).reshape(-1,1).astype(float)
    if len(pts) < k: return np.sort(np.linspace(0,255,k))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(pts)
    return np.sort(km.cluster_centers_.flatten())

def clean_mask(mask):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    if n <= 1: return mask
    areas = stats[1:,cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    out = np.zeros_like(mask)
    out[lbl==largest]=255
    return out

def bbox(mask):
    pts = cv2.goodFeaturesToTrack(mask,10,0.05,50)
    if pts is None: return 0,0,(0,0),(0,0)
    pts = np.int32(pts)
    x1,y1 = np.min(pts[:,:,0]), np.min(pts[:,:,1])
    x2,y2 = np.max(pts[:,:,0]), np.max(pts[:,:,1])
    return x2-x1, y2-y1, (x1,y1), (x2,y2)

def to_mm(dx,dy,px,py,camh=300,f=6.5,viewport=(6.144,8.6)):
    tx=(dx/px)*viewport[1]; ty=(dy/py)*viewport[0]
    return (0.82*(camh/f)*tx,0.79*(camh/f)*ty)

def mean_depth(d, tl, br):
    lx,ly=tl; rx,ry=br
    ly=max(0,ly); ry=min(d.shape[0],ry)
    lx=max(0,lx); rx=min(d.shape[1],rx)
    if ry<=ly or rx<=lx: return float(d.mean())
    return float(np.mean(d[ly:ry,lx:rx]))

# ----------------- UI -----------------
with st.expander("Input", expanded=True):
    file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    camh = st.number_input("Camera height (mm)",300)
    ref_h = st.number_input("Reference object height (mm)",50)
    objs = st.number_input("No. of objects",1,min_value=1)
    run = st.button("Run Measurement")

if run and file:
    st.info("Running...")
    image = Image.open(file).convert("RGB")
    arr = np.array(image)
    proc, model, device = load_model()

    inputs = proc(images=image, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad(): out = model(**inputs)
    res = proc.post_process_depth_estimation(out, target_sizes=[(image.height,image.width)])[0]
    depth = res[list(res.keys())[0]].squeeze().cpu().numpy()

    dn = (depth-depth.min())/(depth.max()-depth.min()+1e-8)
    dg = (dn*255).astype(np.uint8)
    magma = plt.cm.get_cmap("magma")
    dc = (magma(dn)[:,:,:3]*255).astype(np.uint8)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(dc,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    smooth = gaussian_filter1d(hist,1.89)

    # minima in histogram
    d = np.gradient(smooth)
    z = np.where(np.diff(np.sign(d)))[0]
    mins = np.array([i for i in z if d[i-1]<0 and d[i+1]>0])
    if len(mins)==0: mins=[np.argmin(smooth)]

    # DoG
    s1,s2=3.76,1.8
    sm1=gaussian_filter1d(hist,s1); sm2=gaussian_filter1d(hist,s2)
    dog=sm1-sm2; smoothdog=gaussian_filter1d(dog,1.5)
    dd=np.gradient(smoothdog); z2=np.where(np.diff(np.sign(dd)))[0]
    mins2=np.array([i for i in z2 if dd[i-1]<0 and dd[i+1]>0])
    if len(mins2)==0: mins2=[np.argmin(smoothdog)]

    c1=kmeans_centers(mins,int(objs))
    c2=kmeans_centers(mins2,int(objs))
    centers=np.sort((c1+c2)/2)

    gval=int(min(mins[0],mins2[0])) if len(mins)>0 and len(mins2)>0 else int(np.argmin(smooth))
    _,ground=cv2.threshold(gray,gval,255,cv2.THRESH_BINARY)

    masks={}
    if objs>1:
        for i in range(1,int(objs)):
            thr=int(centers[i]) if i<len(centers) else int(centers[-1])
            _,th=cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
            bin=cv2.subtract(ground,th)
            masks[i]=clean_mask(bin)
        summ=np.zeros_like(gray)
        for i in range(1,int(objs)): summ=cv2.add(summ,masks[i])
        resi=cv2.subtract(ground,summ); _,resi=cv2.threshold(resi,1,255,cv2.THRESH_BINARY)
        masks[0]=clean_mask(resi)
    else:
        masks[0]=clean_mask(ground); resi=np.zeros_like(gray)

    # measurements
    temp=dc.copy(); results=[]
    boxes=[]
    for i in range(int(objs)):
        m=masks[i]; dx,dy,tl,br=bbox(m)
        xmm,ymm=to_mm(dx,dy,image.height,image.width,camh)
        boxes.append([tl,br])
        cv2.rectangle(temp,tl,br,(0,255,0),2)
        cv2.putText(temp,f"W {int(xmm)}mm",(tl[0],br[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(temp,f"L {int(ymm)}mm",(tl[0]+5,tl[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        results.append({"Object":i+1,"Width (mm)":int(xmm),"Length (mm)":int(ymm)})

    ref=mean_depth(dn,(0,0),boxes[0][0])
    vals=[]
    for i in range(int(objs)):
        maskbool=(masks[i]>0)
        meanv=float(np.mean(dn[maskbool])) if np.any(maskbool) else float(dn.mean())
        vals.append(meanv)
    near=min([v for v in vals if v>ref], default=ref+1)
    scale=(near-ref) if (near-ref)!=0 else 1
    for i in range(int(objs)):
        depthmm=(vals[i]-ref)/scale*ref_h
        cv2.putText(temp,f"D {int(depthmm)}mm",(boxes[i][0][0],boxes[i][0][1]-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        results[i]["Depth (mm)"]=int(depthmm)

    show_img(temp,"Final Annotated Output")
    st.dataframe(pd.DataFrame(results))
else:
    st.info("Upload an image, set inputs, then press Run.")
