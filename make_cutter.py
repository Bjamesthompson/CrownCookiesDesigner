#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import scale as shapely_scale
import trimesh

def load_and_threshold(png_path):
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open {png_path}")
    _, bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return bw

def find_contours(bw):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        pts = cnt.squeeze().astype(float)
        if pts.ndim == 2 and len(pts) >= 3:
            polys.append(Polygon(pts))
    return unary_union(polys)

def normalize(poly, inside_target):
    minx, miny, maxx, maxy = poly.bounds
    scale_factor = inside_target / max(maxx-minx, maxy-miny)
    return shapely_scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(minx, miny))

def build_cutter(poly, wall, total_height, cut_thickness, split_ratio, base_offset, base_height):
    meshes = []

    # 1) Hollow base flange
    base_outer = poly.buffer(base_offset)
    base_ring  = base_outer.difference(poly)
    base_mesh  = trimesh.creation.extrude_polygon(base_ring, base_height)
    meshes.append(base_mesh)

    # 2) Thick wall section (split_ratio of total_height)
    h1 = total_height * split_ratio
    outer_thick = poly.buffer(wall)
    ring_thick  = outer_thick.difference(poly)
    mesh1 = trimesh.creation.extrude_polygon(ring_thick, h1)
    mesh1.apply_translation((0, 0, base_height))
    meshes.append(mesh1)

    # 3) Thin cutting edge (remaining height)
    h2 = total_height - h1
    outer_thin = poly.buffer(cut_thickness)
    ring_thin  = outer_thin.difference(poly)
    mesh2 = trimesh.creation.extrude_polygon(ring_thin, h2)
    mesh2.apply_translation((0, 0, base_height + h1))
    meshes.append(mesh2)

    # Combine into one mesh
    return trimesh.util.concatenate(meshes)

def main(png_path, name,
         wall=2.0, inside=100.0,
         total_height=10.0, cut_thickness=1.0,
         split_ratio=0.75, base_offset=10.0, base_height=5.0):
    png = Path(png_path)
    if not png.exists():
        print(f"❌ File not found: {png}")
        sys.exit(1)

    bw = load_and_threshold(png)
    outline = find_contours(bw)
    inner = normalize(outline, inside)

    mesh = build_cutter(inner,
                        wall,
                        total_height,
                        cut_thickness,
                        split_ratio,
                        base_offset,
                        base_height)

    out_stl = Path(f"{name}.stl")
    mesh.export(out_stl)
    print(f"✅ Generated {out_stl}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Make cookie-cutter STL from PNG with hollow base + 2-stage wall"
    )
    p.add_argument("png",    help="input PNG file")
    p.add_argument("name",   help="output base name (no extension)")
    p.add_argument("--wall",          type=float, default=2.0,
                   help="thick wall thickness (mm)")
    p.add_argument("--inside",        type=float, default=100.0,
                   help="inside max dimension (mm)")
    p.add_argument("--height",        type=float, default=10.0,
                   help="total height ABOVE base (mm)")
    p.add_argument("--cut_thickness", type=float, default=1.0,
                   help="thin edge thickness (mm)")
    p.add_argument("--split_ratio",   type=float, default=0.75,
                   help="fraction of wall height that is thick (0–1)")
    p.add_argument("--base_offset",   type=float, default=10.0,
                   help="base flange extension (mm)")
    p.add_argument("--base_height",   type=float, default=5.0,
                   help="base flange height (mm)")
    args = p.parse_args()

    main(args.png,
         args.name,
         args.wall,
         args.inside,
         args.height,
         args.cut_thickness,
         args.split_ratio,
         args.base_offset,
         args.base_height)
