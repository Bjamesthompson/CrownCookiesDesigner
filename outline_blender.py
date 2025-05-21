```python
# ultimate_cookie_cutter_app.py

import os
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale
import trimesh
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev

# ————— PAGE CONFIG —————————————————————————————————————
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("🍪 Ultimate Outline & Cookie Cutter App")

# ————— SHAPELY HELPERS ———————————————————————————————————
def ensure_poly(g):
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda p: p.area)
    return g

def smooth_polygon(poly, iters):
    coords = list(poly.exterior.coords)[:-1]
    for _ in range(iters):
        new = []
        n = len(coords)
        for i in range(n):
            p0 = np.array(coords[i])
            p1 = np.array(coords[(i+1) % n])
            new.append(tuple(0.75*p0 + 0.25*p1))
            new.append(tuple(0.25*p0 + 0.75*p1))
        coords = new
    return Polygon(coords)

def densify_shapely(poly, spacing):
    line = LineString(poly.exterior.coords)
    length = line.length
    steps  = max(int(length/spacing),1)
    pts    = [line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0,1,steps+1)]
    return Polygon(pts)

# ————— MESH CREATOR —————————————————————————————————————————
def make_mesh(inner_poly):
    smooth = smooth_polygon(inner_poly, SMOOTH_ITERS)
    dense  = densify_shapely(smooth, DENSIFY_SPACING)
    parts  = []

    flange = dense.buffer(base_off, resolution=BUFFER_RES).difference(dense)
    parts.append(trimesh.creation.extrude_polygon(flange, base_h))

    h1   = height * split_frac
    ring = dense.buffer(wall, resolution=BUFFER_RES).difference(dense)
    m1   = trimesh.creation.extrude_polygon(ring, h1)
    m1.apply_translation((0,0,base_h)); parts.append(m1)

    ring2 = dense.buffer(cut_edge, resolution=BUFFER_RES).difference(dense)
    m2    = trimesh.creation.extrude_polygon(ring2, height-h1)
    m2.apply_translation((0,0,base_h+h1)); parts.append(m2)

    return trimesh.util.concatenate(parts)

# ————— DENSIFY PIXEL CONTOUR —————————————————————————————————
def densify_pixels(coords, max_dist=1.0):
    pts = []
    for (x0,y0),(x1,y1) in zip(coords, coords[1:]):
        pts.append((x0,y0))
        dx, dy = x1-x0, y1-y0
        dist = np.hypot(dx,dy)
        if dist > max_dist:
            n = int(np.ceil(dist/max_dist))
            for i in range(1,n):
                pts.append((x0+dx*i/n, y0+dy*i/n))
    pts.append(coords[-1])
    return pts

# ————— UI TABS —————————————————————————————————————————
tab1, tab2 = st.tabs(["🖌️ Outline Editor", "🍪 Cutter Generator"])

with tab1:
    st.header("🔍 Outline Editor")
    uploaded    = st.file_uploader("Upload image (PNG/JPG)", type=["png","jpg","jpeg"])
    thresh      = st.slider("Mask threshold", 0,255,240)
    expand_p    = st.slider("Outline expand (px)", 0,200,20)
    line_w      = st.slider("Line width (px)", 1,20,3)
    spline_s    = st.slider("Spline smoothness (s)", 0.0,10000.0,0.0, step=100.0)

    if uploaded:
        data     = np.frombuffer(uploaded.read(), np.uint8)
        orig_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        h, w     = orig_bgr.shape[:2]
        gray     = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea).squeeze().astype(float)
            poly  = Polygon(c)
            poly  = ensure_poly(poly)
            buf   = poly.buffer(expand_p)
            raw_coords = list(buf.exterior.coords)
            pts   = densify_pixels(raw_coords, max_dist=1.0)
            if spline_s>0 and len(pts)>4:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                try:
                    tck,u = splprep([xs,ys], s=spline_s, per=True)
                    unew  = np.linspace(0,1,len(pts))
                    xs2, ys2 = splev(unew, tck)
                    outline_pts = list(zip(xs2,ys2))
                except:
                    outline_pts = pts
            else:
                outline_pts = pts
            st.session_state['outline_pts'] = outline_pts

            margin   = int(expand_p + line_w + 5)
            canvas_w = w+2*margin; canvas_h = h+2*margin
            canvas1  = cv2.copyMakeBorder(orig_bgr, margin,margin,margin,margin,
                                          cv2.BORDER_CONSTANT, value=(255,255,255))
            canvas2  = np.ones((canvas_h,canvas_w,3),dtype=np.uint8)*255
            pts_arr  = np.array(outline_pts, dtype=np.int32).reshape(-1,1,2)+margin
            cv2.polylines(canvas1,[pts_arr],True,(255,0,0),thickness=line_w,lineType=cv2.LINE_AA)
            cv2.polylines(canvas2,[pts_arr],True,(0,0,0),thickness=line_w,lineType=cv2.LINE_AA)
            img1 = Image.fromarray(cv2.cvtColor(canvas1,cv2.COLOR_BGR2RGB))
            img2 = Image.fromarray(canvas2)
            c1,c2 = st.columns(2)
            with c1: st.image(img1, use_container_width=True, caption="Overlay")
            with c2: st.image(img2, use_container_width=True, caption="Outline Only")
        else:
            st.error("No silhouette found—adjust threshold.")

with tab2:
    st.header("🍪 Cutter Generator")
    if 'outline_pts' not in st.session_state:
        st.warning("First generate an outline in the Outline Editor tab.")
    else:
        out_name     = st.text_input("Filename (no ext)","cookie_cutter")
        inside       = st.slider("Inside span (mm)",10,200,100)
        wall         = st.slider("Wall thickness (mm)",1,10,2)
        height       = st.slider("Wall height (mm)",5,50,10)
        cut_edge     = st.slider("Cut-edge thick (mm)",0,5,1)
        split_frac   = st.slider("Thick-wall % height",0.1,1.0,0.75)
        base_off     = st.slider("Base offset (mm)",0,50,10)
        base_h       = st.slider("Base height (mm)",0,20,5)
        gauss_k      = st.slider("Mask blur (px)",0,15,5,step=2)
        simplify_tol = st.slider("Simplify tol (px)",0.0,5.0,1.0,step=0.5)
        SMOOTH_ITERS = st.slider("Chaikin iters",0,10,4)
        DENSIFY_SPACING = st.slider("Resample spacing (mm)",0.05,1.0,0.1,step=0.05)
        BUFFER_RES   = st.slider("Buffer resolution (seg/quad)",8,128,64,step=8)
        WATCH_FOLDER = os.getenv("BAMBU_WATCH_FOLDER")

        pts = st.session_state['outline_pts']
        poly = Polygon(pts)
        poly = ensure_poly(poly)
        minx,miny,maxx,maxy = poly.bounds
        sf = inside / max(maxx-minx, maxy-miny)
        inner = shapely_scale(poly, xfact=sf, yfact=sf, origin=(minx,miny))

        mesh = make_mesh(inner)
        v,f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f[:,0], j=f[:,1], k=f[:,2],
            color='lightsteelblue', opacity=0.6
        )])
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(aspectmode='data'))
        st.plotly_chart(fig, use_container_width=True)

        tmp = Path(tempfile.NamedTemporaryFile(suffix='.3mf', delete=False).name)
        mesh.export(tmp)
        data = tmp.read_bytes()
        st.download_button("💾 Save 3MF", data=data, file_name=f"{out_name}.3mf")
        if WATCH_FOLDER:
            dst = Path(WATCH_FOLDER)/f"{out_name}.3mf"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp, dst)
            st.caption(f"✅ Copied to {dst}")
```
