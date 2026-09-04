import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

HERE = Path(__file__).resolve().parent

S = 4
W = 2
res = 200
resC= 200j
INF = 1000

@dataclass
class Sphere:
  center: np.ndarray
  radius: float
  color: np.ndarray


def sphere_test(dir, view_origin, center, radius, pass_depth):
  m = view_origin - center
  b = 2 * np.sum(m * dir, axis=-1)
  c = np.sum(m * m, axis=-1) - radius**2
  discriminant = b**2 - 4 * c
  #disc < 0 means that the sphere is not hit, return pass_depth
  root = np.sqrt(np.where(discriminant < 0, 0.0, discriminant))
  tested_depth = np.where(discriminant < 0, pass_depth, (-b - root) / 2.0)
  return np.where(tested_depth > pass_depth, pass_depth, tested_depth)


#objects
orb = Sphere(np.array([2,0,0]), 1, np.array([0.3, 0.4, 1.0]))
orb2 = Sphere(np.array([1,-.5,-.5]), .5, np.array([1.0, 0.4, 0.3]))

#light direction
l = np.array([-1.0, .3, -.4])
l = l / np.linalg.norm(l)

#view coords
u, v = np.mgrid[-W:W:S*resC, -W:W:S*resC]

#in perspective
#view vector
d = np.zeros((res * S, res * S, 3), dtype=np.float64)
d[...,0] = 1
d[...,1] = 0*u
d[...,2] = 0*v
d = d / np.linalg.norm(d, axis=-1, keepdims=True)
#view origin
vo = np.zeros((res * S, res * S, 3), dtype=np.float64)
vo[...,0] = 0
vo[...,1] = 1*u
vo[...,2] = 1*v


#the depth buffer
depth = np.ones((res * S, res * S), dtype=np.float64)
#begin with infinite depth
depth = depth * INF

#coloring and background
c = np.ones((res * S, res * S, 3), dtype=np.float64)
c = c * np.array([1.0, 0.85, 0.6])

#normals +x points into screen
n = np.ones((res * S, res * S, 3), dtype=np.float64)
n = n * l

#depth testing / passes
t = sphere_test(d, vo, orb.center, orb.radius, depth)
depth_test = t < depth
c[depth_test] = orb.color
p = vo + t[..., None] * d
normal = (p - orb.center) / orb.radius
n[depth_test] = normal[depth_test]
depth[depth_test] = t[depth_test]

t = sphere_test(d, vo, orb2.center, orb2.radius, depth)
depth_test = t < depth
c[depth_test] = orb2.color
p = vo + t[..., None] * d
normal = (p - orb2.center) / orb2.radius
n[depth_test] =  normal[depth_test]
depth[depth_test] = t[depth_test]

#light
s = n @ l

c[...,0] = c[...,0] * s
c[...,1] = c[...,1] * s
c[...,2] = c[...,2] * s


#c = np.ones((res * S, res * S, 3), dtype=np.float64)
#c = c * 1 / (depth[..., None]+1)


#final cleaning
o = c * 255
o = np.clip(o, 0 ,255)

i = Image.fromarray(o.astype(np.uint8))
i.resize((res, res), Image.LANCZOS).save(HERE / "first_depth.png")