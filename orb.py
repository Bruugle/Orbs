import numpy as np
from PIL import Image
from pathlib import Path

HERE = Path(__file__).resolve().parent

S = 4
W = 2
res = 500
resC= 500j

radius = 1

y, x = np.mgrid[-W:W:S*resC, -W:W:S*resC]
r = np.sqrt(x**2 + y**2) ## distance from center

a = np.ones((res * S, res * S, 3), dtype=np.float64)


#normals
n = a * np.array([0,0,1])
n[...,0] = x
n[...,1] = y
n[...,2] = np.sqrt(np.abs(1 - x**2 - y**2))
m = np.heaviside(r - radius, .5)
n[r>1] = np.array([0,0,1])



# colors
a[...,0] = m
a[...,1] = m
a[...,2] = m


b = (1 - a) * np.array([0.3, 0.4, 1.0])

a = a * np.array([1.0, 0.85, 0.6])
a = a + b

#light direction
l = np.array([-0.3, 0.4, 1.0])
l = l / np.linalg.norm(l)
s = n @ l

#light mixing

a[...,0] = a[...,0] * s
a[...,1] = a[...,1] * s
a[...,2] = a[...,2] * s


a = a * 255
a = np.clip(a, 0 ,255)

i = Image.fromarray(a.astype(np.uint8))
i.resize((res, res), Image.LANCZOS).save(HERE / "first_pure_orb.png")