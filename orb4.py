import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

HERE = Path(__file__).resolve().parent

S = 4
W = 2
res = 600
resC= 600j
INF = 1000
data_type = np.float32

@dataclass
class Sphere:
  center: np.ndarray
  radius: float
  color: np.ndarray

@dataclass
class Rect:
  pivot: np.ndarray
  axis1: np.ndarray
  axis2: np.ndarray
  color: np.ndarray



def sphere_test(dir, view_origin, center, radius, pass_depth):
  m = view_origin - center
  b = 2 * np.sum(m * dir, axis=-1)
  c = np.sum(m * m, axis=-1) - radius**2
  discriminant = b**2 - 4 * c
  #disc < 0 means that the sphere is not hit, return pass_depth
  root = np.sqrt(np.where(discriminant < 0, 0.0, discriminant))
  fail = (discriminant < 0) | (root < 1e-4)
  tested_depth = np.where(fail, pass_depth, (-b - root) / 2.0)
  fail = (tested_depth > pass_depth) | (tested_depth < 1e-4)
  return np.where(fail, pass_depth, tested_depth)

def striped_sphere_test(axis, scale, ratio ,dir, view_origin, center, radius, pass_depth):
  m = view_origin - center
  b = 2 * np.sum(m * dir, axis=-1)
  c = np.sum(m * m, axis=-1) - radius**2
  discriminant = b**2 - 4 * c
  #disc < 0 means that the sphere is not hit, return pass_depth
  root = np.sqrt(np.where(discriminant < 0, 0.0, discriminant))
  tested_depth = np.where(discriminant < 0, pass_depth, (-b - root) / 2.0)
  
  test_point1 = view_origin + tested_depth[...,None] * dir
  test_v1 = test_point1 - center
  proj1 = np.sum(axis * test_v1, axis=-1)
  

  fail1 = (tested_depth > pass_depth) | (tested_depth < 1e-4) | (proj1 * scale % 1. < ratio)
  current_depth = np.where(fail1, pass_depth, tested_depth)

  tested_back_depth = np.where(discriminant < 0, current_depth, (-b + root) / 2.0)
  test_point2 = view_origin + tested_back_depth[...,None] * dir
  test_v2 = test_point2 - center
  proj2 = np.sum(axis * test_v2, axis=-1)

  fail2 = (tested_back_depth > current_depth) | (tested_back_depth < 1e-4) | (proj2 * scale % 1. < ratio)

  return np.where(fail2, current_depth, tested_back_depth)
   

def rect_test(dir, view_origin, pivot, axis1, axis2, pass_depth):
  norm = np.cross(axis1, axis2)
  denom = np.sum(norm * dir, axis=-1)
  not_parallel = np.abs(denom) > 1e-9 # for the inner product equation below
  t = np.sum(norm * (pivot - view_origin), axis=-1) / np.where(not_parallel, denom, 1.0)
  hit = not_parallel & (t > 1e-4) # in front of view
  p = view_origin + t[...,None] * dir #the point of intersect
  w = p - pivot #setting to origin to check offsets
  a = np.sum(w * axis1, axis=-1) / (axis1 @ axis1)
  b = np.sum(w * axis2, axis=-1) / (axis2 @ axis2)
  hit = hit & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
  return np.where(hit, t, pass_depth)



#objects
orb = Sphere(np.array([2,0,0]), 1, np.array([0, 0.5, .55]))
orb1 = Sphere(np.array([2,0,0]), 1.1, np.array([0, 0.5, .55]))
orb2 = Sphere(np.array([2,0,0]), .8, np.array([0, .8, 0.8]))
orb3 = Sphere(np.array([2,0,0]), 1.1, np.array([.5, .8, .9]))
#plane = Rect(np.array([2,1.5,-3]), np.array([6,-.2,.1]), np.array([0,-.2,6]), np.array([1.0, 0.85, 0.6]))

#light direction
l = np.array([-.5, -.4, -.3])
l = l / np.linalg.norm(l)

#ambient light amount between 0-1
amb =  0.15

#view coords
u, v = np.mgrid[-W:W:S*res*1j, -W:W:S*res*1j]

#jitter
step = 2.0 * W / (res * S)
rng = np.random.default_rng(0)
u = u + rng.uniform(-0.5, 0.5, u.shape) * step
v = v + rng.uniform(-0.5, 0.5, v.shape) * step

#in perspective
#view vector
d = np.zeros((res * S, res * S, 3), dtype=data_type)
d[...,0] = 1
d[...,1] = .2*u
d[...,2] = .2*v
d = d / np.linalg.norm(d, axis=-1, keepdims=True)
#view origin
vo = np.zeros((res * S, res * S, 3), dtype=data_type)
vo[...,0] = 0
vo[...,1] = .8*u
vo[...,2] = .8*v


#the depth buffer
depth = np.ones((res * S, res * S), dtype=data_type)
#begin with infinite depth
depth = depth * INF

#coloring and background
c = np.ones((res * S, res * S, 3), dtype=data_type)
c = c * np.array([0,0,0])

#normals +x points into screen
n = np.ones((res * S, res * S, 3), dtype=data_type)
n = n * l



#depth testing / passes
t = striped_sphere_test(np.array([-1,1,1]), 1.7, .3 ,d, vo, orb.center, orb.radius, depth)
depth_test = t < depth
c[depth_test] = orb.color
p = vo + t[..., None] * d
normal = (p - orb.center) / orb.radius
back = np.sum(normal * d, axis=-1) > 0
normal = np.where(back[..., None], -normal, normal)
n[depth_test] = normal[depth_test]
depth[depth_test] = t[depth_test]


t = striped_sphere_test(np.array([1,-1,1]), 2, .5 ,d, vo, orb1.center, orb1.radius, depth)
depth_test = t < depth
c[depth_test] = orb1.color
p = vo + t[..., None] * d
normal = (p - orb1.center) / orb1.radius
back = np.sum(normal * d, axis=-1) > 0
normal = np.where(back[..., None], -normal, normal)
n[depth_test] = normal[depth_test]
depth[depth_test] = t[depth_test]


t = striped_sphere_test(np.array([0,-2,.5]), 2, .4 ,d, vo, orb2.center, orb2.radius, depth)
depth_test = t < depth
c[depth_test] = orb2.color
p = vo + t[..., None] * d
normal = (p - orb2.center) / orb2.radius
back = np.sum(normal * d, axis=-1) > 0
normal = np.where(back[..., None], -normal, normal)
n[depth_test] =  normal[depth_test]
depth[depth_test] = t[depth_test]


t = sphere_test(d, vo, orb3.center, orb3.radius, depth)
depth_test = t < depth
c[depth_test] = orb3.color
p = vo + t[..., None] * d
normal = (p - orb3.center) / orb3.radius
back = np.sum(normal * d, axis=-1) > 0
normal = np.where(back[..., None], -normal, normal)
n[depth_test] =  normal[depth_test]
depth[depth_test] = t[depth_test]


#shadows
#the sahdow depth buffer
s_depth = np.ones((res * S, res * S), dtype=data_type)
#begin with infinite depth and the shadow test point
s_depth = s_depth * INF
s_origin = vo + depth[...,None] * d #+ 1e-4 * n



t = striped_sphere_test(np.array([-1,1,1]), 1.7, .3 ,l, s_origin, orb.center, orb.radius, s_depth)
depth_test = t < s_depth
s_depth[depth_test] = t[depth_test]

t = striped_sphere_test(np.array([1,-1,1]), 2, .5 ,l, s_origin, orb1.center, orb1.radius, s_depth)
depth_test = t < s_depth
s_depth[depth_test] = t[depth_test]


t = striped_sphere_test(np.array([0,-2,.5]), 2, .4 ,l, s_origin, orb2.center, orb2.radius, s_depth)
depth_test = t < s_depth
s_depth[depth_test] = t[depth_test]


t = sphere_test(l, s_origin, orb3.center, orb3.radius, s_depth)
depth_test = t < s_depth
s_depth[depth_test] = t[depth_test]



#shadow clipping
s = np.where(s_depth < INF, 0, 1)

#lighting - lambert
lam = np.clip(np.sum(n * l, axis=-1), 0, 1)


##blinn-phong
ks = .5
shininess = 60
h = l - d
h = h / np.linalg.norm(h, axis=-1, keepdims=True)
ndoth = np.sum(n * h, axis=-1)
spec = np.where((lam > 0) & (s > 0), np.clip(ndoth, 0, 1) ** shininess, 0.0)


c = c = c * np.clip(s * lam, amb, 1.0)[..., None] + ks * spec[..., None] / depth[...,None]

#c = np.ones((res * S, res * S, 3), dtype=np.float64)
#c = c * 1 / (depth[..., None]+1)


#final cleaning
o = c * 255
o = np.clip(o, 0 ,255)

i = Image.fromarray(o.astype(np.uint8))
i.resize((res, res), Image.LANCZOS).save(HERE / "magnum_orbus_hashed.png")