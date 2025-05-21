import os
import tempfile
import shutil

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale
import trimesh
import plotly.graph_objects as go

# ————— PAGE CONFIG —————————————————————————————————————
st.set_page_config(layout="wide")
st.title("🍪 Smoothed Cookie-Cutter Generator")

# ————— HELPERS —————————————————————————————————————
def ensure_poly(g):
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda p: p.area)
    return g

def load_mask(data: bytes, gauss_k: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gauss_k > 1:
        k = gauss_k if gauss_k % 2 else gauss_k + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    _, bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return bw

def extract_polygon(bw: np.ndarray, simplify_tol: float) -> Polygon:
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        pts = c.squeeze().astype(float)
        if pts.ndim == 2 and len(pts) >= 3:
            polys.append(Polygon(pts))
    outline = unary_union(polys)
    if simplify_tol > 0:
        outline = outline.simplify(simplify_tol, preserve_topology=True)
    return ensure_poly(outline)

def chaikin_points(coords, iters):
    for _ in range(iters):
        new = []
        n = len(coords)
        for i in range(n):
            p0 = np.array(coords[i])
            p1 = np.array(coords[(i+1) % n])
            new.append(tuple(0.75*p0 + 0.25*p1))
            new.append(tuple(0.25*p0 + 0.75*p1))
        coords = new
    return coords

def smooth_polygon(poly, iters):
    ring = list(poly.exterior.coords)[:-1]
    return Polygon(chaikin_points(ring, iters))

def densify(poly, spacing):
    line = LineString(poly.exterior.coords)
    steps = max(int(line.length / spacing), 1)
    pts = [
        line.interpolate(t, normalized=True).coords[0]
        for t in np.linspace(0, 1, steps+1)
    ]
    return Polygon(pts)

def make_mesh(inner_poly):
    # 1) Smooth with Chaikin
    smooth = smooth_polygon(inner_poly, SMOOTH_ITERS)
    # 2) Resample for nice offsets
    dense  = densify(smooth, DENSIFY_SPACING)
    parts  = []

    # 3) Base flange
    flange = dense.buffer(base_off, resolution=BUFFER_RES).difference(dense)
    parts.append(trimesh.creation.extrude_polygon(flange, base_h))

    # 4) Thick wall
    h1   = height * split_frac
    ring = dense.buffer(wall, resolution=BUFFER_RES).difference(dense)
    m1   = trimesh.creation.extrude_polygon(ring, h1)
    m1.apply_translation((0, 0, base_h))
    parts.append(m1)

    # 5) Cutting edge
    ring2 = dense.buffer(cut_edge, resolution=BUFFER_RES).difference(dense)
    m2    = trimesh.creation.extrude_polygon(ring2, height - h1)
    m2.apply_translation((0, 0, base_h + h1))
    parts.append(m2)

    # 6) Combine & return
    mesh = trimesh.util.concatenate(parts)
    return mesh

# ————— TWO-COLUMN LAYOUT —————————————————————————————————————
col_ctrl, col_view = st.columns([1, 3], gap="small")

with col_ctrl:
    st.subheader("⚙️ Controls")
    png_file      = st.file_uploader("Upload (PNG/JPG)", type=["png","jpg","jpeg"])
    out_name      = st.text_input("Filename (no ext)", "cookie_cutter")

    inside        = st.slider("Inside span (mm)",            10, 200, 100)
    wall          = st.slider("Wall thickness (mm)",         1,  10,   2)
    height        = st.slider("Wall height (mm)",            5,  50,  10)
    cut_edge      = st.slider("Cut-edge thick (mm)",          0,   5,   1)

    split_frac    = st.slider("Thick-wall % height",       0.1, 1.0, 0.75)
    base_off      = st.slider("Base offset (mm)",            0,  50,   4)
    base_h        = st.slider("Base height (mm)",            0,  20,    5)

    gauss_k       = st.slider("Mask blur (px)",              0,  15,    5, step=2)
    simplify_tol  = st.slider("Simplify tol (px)",         0.0,  5.0,  1.0, step=0.5)
    SMOOTH_ITERS  = st.slider("Chaikin iters",               0,  10,    6)
    DENSIFY_SPACING = st.slider("Resample spacing (mm)",   0.05, 1.00,  0.10, step=0.05)
    BUFFER_RES    = st.slider("Buffer resolution (seg/quad)", 8, 128,   128, step=8)

    WATCH_FOLDER = os.getenv("BAMBU_WATCH_FOLDER")
    if not WATCH_FOLDER:
        st.caption("Set BAMBU_WATCH_FOLDER to auto-copy your .3mf")

with col_view:
    if not png_file:
        st.info("Upload a PNG/JPG on the left to see your cutter here.")
    else:
        # 1) Load & clean mask
        bw      = load_mask(png_file.read(), gauss_k)
        # 2) Extract + simplify polygon
        outline = extract_polygon(bw, simplify_tol)
        # 3) Scale to desired inside span
        minx, miny, maxx, maxy = outline.bounds
        sf      = inside / max(maxx - minx, maxy - miny)
        inner   = shapely_scale(outline, xfact=sf, yfact=sf,
                                origin=(minx, miny))

        # 4) Build mesh & preview
        mesh = make_mesh(inner)
        v, f = mesh.vertices, mesh.faces
        fig = go.Figure(data=[go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f[:,0], j=f[:,1], k=f[:,2],
            color='lightsteelblue', opacity=0.6
        )])
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode='data')
        )
        st.subheader("🔍 3D Preview")
        st.plotly_chart(fig, use_container_width=True)

        # 5) Download & copy buttons
        tmp = Path(tempfile.NamedTemporaryFile(
            suffix=".3mf", delete=False).name)
        mesh.export(tmp)
        data = tmp.read_bytes()
        st.download_button("💾 Save 3MF", data=data,
                           file_name=f"{out_name}.3mf")
        if WATCH_FOLDER:
            dst = Path(WATCH_FOLDER) / f"{out_name}.3mf"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp, dst)
            st.caption(f"✅ Copied to {dst}")
