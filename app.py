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

# ---------------- Input ----------------
with st.expander("Input Parameters", expanded=True):
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    relative_height_ratio = st.selectbox("Relative Height Ratio", ["low","med","high","vhigh"])
    camh = st.number_input("Camera Height (mm)", value=300)
    ref_h = st.number_input("Reference Object Height (mm)", value=50)
    nom_of_objects = st.number_input("Number of Objects", value=2, min_value=1)
    run_process = st.button("Run Measurement")

# ---------------- Helpers ----------------
def small_area_remover(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    if n>1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        out[labels==np.argmax(areas)+1]=255
    return out

def find_local_minima(a):
    g=np.gradient(a)
    return np.where((np.concatenate(([g[0]],g[:-1]))<0)&(g>0))[0]

def safe_kmeans_centers(points,n,low=0,high=255):
    if points is None or len(points)==0:
        return np.linspace(low,high,n+1)[1:]
    pts=np.array(points).reshape(-1,1).astype(float)
    if pts.shape[0]<n:
        extra_needed=n-pts.shape[0]
        extra=np.linspace(low,high,extra_needed+2,dtype=int)[1:-1]
        if extra.size>0: pts=np.vstack([pts,extra.reshape(-1,1)])
    k=KMeans(n_clusters=n,random_state=42).fit(pts)
    return np.sort(k.cluster_centers_.reshape(-1))

def centered_visual(img, caption=None, width=550):
    if isinstance(img,np.ndarray):
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    b=io.BytesIO(); img.save(b,"PNG")
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;margin-bottom:40px;'>
      <img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"
           style='width:{width}px;border-radius:6px;'>
      <div style='text-align:left;width:{width}px;margin-top:6px;'>
        <p style='font-size:16px;font-weight:600;'>{caption or ''}</p>
      </div>
    </div>""",unsafe_allow_html=True)

@st.cache_resource
def load_depth_model():
    mid="depth-anything/Depth-Anything-V2-Small-hf"
    proc=AutoImageProcessor.from_pretrained(mid)
    mdl=AutoModelForDepthEstimation.from_pretrained(mid)
    return proc,mdl

# ---------- improved vertical text ----------
def vertical_text(img, text, org, color=(255,255,0), angle=90):
    """Fully visible vertical text (rotated phrase with padding + alpha blend)."""
    font=cv2.FONT_HERSHEY_SIMPLEX; scale,thick=1,3
    (tw,th),_=cv2.getTextSize(text,font,scale,thick)

    pad=60
    canvas_h,canvas_w=tw+pad*2,th+pad*2
    canvas=np.zeros((canvas_h,canvas_w,4),np.uint8)

    org_x,org_y=pad,canvas_h//2+th//2
    cv2.putText(canvas,text,(org_x,org_y),font,scale,(0,0,0,255),thick+3,cv2.LINE_AA)
    cv2.putText(canvas,text,(org_x,org_y),font,scale,(*color,255),thick,cv2.LINE_AA)

    M=cv2.getRotationMatrix2D((canvas_w/2,canvas_h/2),angle,1.0)
    rot=cv2.warpAffine(canvas,M,(canvas_w,canvas_h),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))

    x,y=org; h,w=rot.shape[:2]
    y=max(0,min(y,img.shape[0]-h))
    x=max(0,min(x,img.shape[1]-w))
    alpha=rot[:,:,3:]/255.0
    roi=img[y:y+h,x:x+w]
    roi[:]=(alpha*rot[:h,:w,:3]+(1-alpha)*roi).astype(np.uint8)
    img[y:y+h,x:x+w]=roi
    return img

# ---------------- Run ----------------
if run_process and uploaded_file:
    st.info("Processing image …")
    image=Image.open(uploaded_file); rgb=np.array(image.convert("RGB"))
    proc,mdl=load_depth_model()
    inp=proc(images=image,return_tensors="pt")
    with torch.no_grad(): out=mdl(**inp)
    d=proc.post_process_depth_estimation(out,target_sizes=[(image.height,image.width)])[0]
    depth=d.get("predicted_depth",d.get("depth")).squeeze().cpu().numpy()

    dn=(depth-depth.min())/(depth.max()-depth.min()+1e-9)
    dg=(dn*255).astype(np.uint8)
    dc=cv2.cvtColor((plt.cm.magma(dn)[:,:,:3]*255).astype(np.uint8),cv2.COLOR_RGB2BGR)

    gray=cv2.cvtColor(dc,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    s1,s2=gaussian_filter1d(hist,3.76),gaussian_filter1d(hist,1.8)
    dog=s1-s2; smooth=1.8*gaussian_filter1d(dog,1.5)
    low={"low":110,"med":100,"high":80,"vhigh":60}[relative_height_ratio]; up=255
    mh=s1[low:up]; min_h=(find_local_minima(mh)+low).astype(int)
    grad=np.gradient(smooth); zc=np.where(np.diff(np.sign(grad)))[0]
    min_d=np.array([i for i in zc if grad[i-1]<0 and grad[i+1]>0],int)
    min_d=min_d[(min_d>=low)&(min_d<up)]
    n=int(max(1,nom_of_objects))
    ch,cd=safe_kmeans_centers(min_h,n,low,up),safe_kmeans_centers(min_d,n,low,up)
    cm=np.sort((np.array(ch)+np.array(cd))/2).astype(int)

    masks={}; thr=int(cm[0]) if len(cm)>0 else low
    _,ground=cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
    if n>1:
        for i in range(1,n):
            thr=int(cm[i]) if i<len(cm) else int(cm[-1])
            _,t=cv2.threshold(gray,thr,255,cv2.THRESH_BINARY)
            masks[i]=small_area_remover(cv2.subtract(ground,t))
        s=np.zeros_like(gray,np.uint8)
        for i in range(1,n): s=cv2.add(s,masks[i])
        res=cv2.subtract(ground,s); _,res=cv2.threshold(res,1,255,cv2.THRESH_BINARY)
        masks[0]=small_area_remover(res)
    else:
        masks[0]=small_area_remover(ground); res=np.zeros_like(gray)

    def sad(mask):
        if mask is None or np.count_nonzero(mask)==0:
            h,w=gray.shape; return w,h,(0,0),(w-1,h-1)
        c=cv2.goodFeaturesToTrack(mask,10,0.05,50)
        if c is None: h,w=gray.shape; return w,h,(0,0),(w-1,h-1)
        c=np.int32(c); x0,y0=np.min(c[:,:,0]),np.min(c[:,:,1])
        x1,y1=np.max(c[:,:,0]),np.max(c[:,:,1])
        return x1-x0,y1-y0,(x0,y0),(x1,y1)

    def view(dx,dy,px,py,camh=300,f=5.42,vp=[6.144,8.6],cx=0.82,cy=0.79):
        return [cx*(camh/f)*(dx/px)*vp[1], cy*(camh/f)*(dy/py)*vp[0]]

    def mean_depth(dm,lt,rb):
        lx,ly=lt; rx,ry=rb
        lx,rx=max(0,lx),min(dm.shape[1]-1,rx)
        ly,ry=max(0,ly),min(dm.shape[0]-1,ry)
        return float(dm.mean()) if ry<=ly or rx<=lx else np.mean(dm[ly:ry,lx:rx])

    temp,res_tab,boxes=dc.copy(),[],[]
    for i in range(n):
        dx,dy,tl,br=sad(masks.get(i))
        x,y=view(dx,dy,rgb.shape[0],rgb.shape[1],camh)
        cv2.rectangle(temp,tl,br,(0,255,0),2)
        boxes.append([tl,br])
        # slightly outside box so always visible
        temp=vertical_text(temp,f"Length {int(y)} mm",(br[0]+15,tl[1]+20),(255,255,0),90)
        cv2.putText(temp,f"Width {int(x)} mm",(tl[0]+10,br[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        res_tab.append({"Object":i+1,"Width (mm)":int(x),"Length (mm)":int(y)})

    ref=mean_depth(dc,(0,0),boxes[0][0]); mvals,min1=[],255
    for i in range(n):
        m=masks[i]//255
        val=dc[m==1].mean() if np.count_nonzero(m) else float(dc.mean())
        if ref<val<min1: min1=val
        mvals.append(val)
    scale=float(min1-ref) if (min1-ref)!=0 else 1.0
    for i in range(n):
        d=(float(mvals[i]-ref)/scale)*ref_h
        cv2.putText(temp,f"Depth {int(d)} mm",boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        res_tab[i]["Depth (mm)"]=int(d)

    # ---- display ----
    st.header("Final Annotated Output")
    centered_visual(temp,"Figure 1. Final annotated image showing Width, Length, and Depth values.")
    st.dataframe(pd.DataFrame(res_tab).style.hide(axis="index"),use_container_width=True)

else:
    if run_process: st.warning("Please upload an image first.")
