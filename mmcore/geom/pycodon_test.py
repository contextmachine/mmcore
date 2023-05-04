import random


def dot(a: list[float], b: list[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def vneg(v: list[float]) -> list[float]:
        a, b, c = v
        return [-a, -b, -c]
def diff(v: list[float], f: list[float]) -> list[float]:
        a, b, c = v
        a1, b1, c1= f
        return [a1-a, b1-b, c1-c]
def vsum(v: list[float], f: list[float],ff: list[float]) -> list[float]:
        a, b, c = v
        a1, b1, c1= f
        a2, b2, c2 = ff
        return [a2+a1+a, b2+b1+b, b2+c1+c]

def vmul(v, f: list[float]) -> list[float]:
        a, b, c = f

        return [v*a, v*b, v*c]

def line_plane_collision(ds, epsilon=1e-6):



    for dt in ds:

        normal, origin, ray_dir, ray_start=dt


        if abs(dot(normal, ray_dir)) < epsilon:
            return None

        return vsum(diff(origin,ray_start), vmul(dot(vneg(normal),diff(origin,ray_start)) / dot(normal, ray_dir), ray_dir), origin)


#dss1=np.random.random((1000,1000,4,3)).tolist()
import time

#s1 = time.time()
#res2 = [py_line_plane_collision(dt) for dt in dss]
#e1 = time.time() - s1
#print(f"py: {e1}")
s=time.time()
res=line_plane_collision([[[random.random(),random.random() ,random.random()]  for i in range(4)] for j in range(1000000)])
e=time.time()-s
print(f"py: {e}")

