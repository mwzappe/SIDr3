#!/usr/bin/env python3

import os
import json
import imageio
from PIL import Image
import numpy as np
import cv2
from sys import exit
from shapely import GeometryCollection, polygons, multipolygons, geometrycollections, wkt,to_geojson
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

Image.MAX_IMAGE_PIXELS = 300000000

img = cv2.imread("MOS_6581R3_full.png", cv2.IMREAD_COLOR)

#print(np.unique(img.reshape(-1, img.shape[2]), axis=0))

#for c in colors:
#    print(f"{c}: {np.where(np.all(img == c, axis=-1))}")

def write_contours(fn, img):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    heirarchy = heirarchy[0]
    
    print(f"Number of contours detected: {len(contours)}")

    assert len(contours) == len(heirarchy)

    resolved = [ False for c in contours ]

    def to_poly(c):
        l = list(map(lambda x: (int(x[0][0]), int(x[0][1])), c))
        try:
            return polygons(l)
        except:
            print(l)
        
    polys = [ to_poly(c) for c in contours ]
    
    def resolve_poly(i):
        if resolved[i]:
            return polys[i]
        
        (n, p, first_child, parent) = heirarchy[i]

        if first_child == -1:
            resolved[i] = True
            return polys[i]

        poly = polys[i]
        
        cur_child = first_child

        while cur_child != -1:
            child_poly = resolve_poly(cur_child)

            poly = poly.difference(child_poly)
            
            cur_child = heirarchy[cur_child][0]

        polys[i] = poly

        resolved[i] = True
        return polys[i]
        

    r = [ resolve_poly(i) for i, _ in enumerate(contours) ]

    out_polys = []

    for i, p in enumerate(polys):
        (_, _, _, parent) = heirarchy[i]
        if parent != -1:
            continue

        out_polys.append(p)

    out_polys = GeometryCollection(out_polys)

    with open(f"{fn}.geojson", "w") as f:
        f.write(to_geojson(out_polys))

    os.system(f"ogr2ogr -f DXF {fn}.dxf {fn}.geojson")
    os.system(f"rm {fn}.geojson")

if False:
    red =  np.zeros(img.shape[:2])
    green = np.zeros(img.shape[:2])
    blue = np.zeros(img.shape[:2])

    red[img[:,:,2] > 1] = 255
    green[img[:,:,1] > 1] = 255
    blue[img[:,:,0] > 1] = 255

    write_contours("red", red)
    write_contours("green", green)
    write_contours("blue", blue)


values = [ 0, 64, 140, 255 ]

colors = [ (b, g, r) for r in values for g in values for b in values ]
    
amax = np.amax(img, axis=-1)

cv2.imwrite("max.png", amax)

if False:
    gray = np.ones(img.shape[:2], dtype='uint8') * 0
    gray[amax == 140] = 255

    cv2.imwrite("gray.png", gray)

    write_contours("gray", gray)

gray = np.ones(img.shape[:2], dtype='uint8') * 0
gray[amax == 64] = 1

cv2.imwrite("darkgray.png", gray)

circle = np.array([
    [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ],
    [ 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
    [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 ],
], dtype="uint8")

conv = convolve2d(gray, circle, "same") 

cv2.imwrite("conv.png", conv)

centers = np.ones(img.shape[:2], dtype='uint8') * 0

s = np.sum(circle)

centers[conv == s] = 255

cv2.imwrite("centers.png", centers)

import ezdxf

doc = ezdxf.new()

msp = doc.modelspace()

for x, y in zip(*np.where(centers == 255)):
    print(x, y)
    msp.add_circle((x, y), 7.5)

doc.saveas("vias.dxf")
