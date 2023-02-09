import numpy as np

def compute_vanishing_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    a = np.array([x1, y1, 1])
    b = np.array([x2, y2, 1])
    c = np.array([x3, y3, 1])
    d = np.array([x4, y4, 1])
    l1 = np.cross(a, b)
    l2 = np.cross(c, d)
    vp = np.cross(l1, l2)
    vp_x, vp_y, _ = vp / vp[2]
    return (vp_x, vp_y)