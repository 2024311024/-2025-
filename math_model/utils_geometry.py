
import math

def vec_add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vec_sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vec_scale(a,s): return (a[0]*s, a[1]*s, a[2]*s)
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm(a): return math.sqrt(dot(a,a))

def unit(vec):
    x,y,z = vec
    L = math.sqrt(x*x+y*y+z*z)
    if L < 1e-15:
        return (0.0, 0.0, 0.0)
    return (x/L, y/L, z/L)

def dist_point_to_segment(Cp, A, B):
    AB = vec_sub(B, A)
    denom = dot(AB, AB)
    if denom < 1e-15:
        return norm(vec_sub(Cp, A)), 0.0
    t = dot(vec_sub(Cp, A), AB) / denom
    lam = max(0.0, min(1.0, t))
    closest = vec_add(A, vec_scale(AB, lam))
    return norm(vec_sub(Cp, closest)), lam

