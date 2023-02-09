# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    a = np.array([points[0][0], points[0][1], 1])
    b = np.array([points[1][0], points[1][1], 1])
    c = np.array([points[2][0], points[2][1], 1])
    d = np.array([points[3][0], points[3][1], 1])
    l1 = np.cross(a, b)
    l2 = np.cross(c, d)
    vp = np.cross(l1, l2)
    vp_x, vp_y, _ = vp / vp[2]
    return [vp_x, vp_y]
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    
    # Define the three vanishing points as column vectors
    v1 = vanishing_points[0]
    v2 = vanishing_points[1]
    v3 = vanishing_points[2]
    
    # Define Matrix A
    A = np.zeros((3,4))
    
    A[0,:] = [(v1[0]*v2[0] + v1[1]*v2[1]), (v1[0]+v2[0]), (v1[1]+v2[1]), 1]
    A[1,:] = [(v1[0]*v3[0] + v1[1]*v3[1]), (v1[0]+v3[0]), (v1[1]+v3[1]), 1]
    A[2,:] = [(v2[0]*v3[0] + v2[1]*v3[1]), (v2[0]+v3[0]), (v2[1]+v3[1]), 1]
    
    # SVD Decomposition of A
    u, s, v = np.linalg.svd(A)
    
    # Get w values from last column of v
    w = v.T[:, -1]
    w1, w4, w5, w6 = w[0], w[1], w[2], w[3]
    
    # Define the omega matrix with given constraints
    omega = np.array([[w1, 0, w4],
                  [0, w1, w5],
                  [w4, w5, w6]])
    
    # Solve for K from omega 
    K = np.linalg.inv(np.linalg.cholesky(omega)).T
    K/=K[2,2]
    
    return K


    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    
    # Compute omega
    omega = np.linalg.inv(np.dot(K, K.T))
    omega_inv = np.linalg.inv(omega)
    
    # Define the two pairs of vanishing points as column vectors
    v1 = np.array([vanishing_pair1[0][0], vanishing_pair1[0][1], 1])
    v2 = np.array([vanishing_pair1[1][0], vanishing_pair1[1][1], 1])
    v3 = np.array([vanishing_pair2[0][0], vanishing_pair2[0][1], 1])
    v4 = np.array([vanishing_pair2[1][0], vanishing_pair2[1][1], 1])

    # Define the two vanishing lines as cross product of the two pairs of vanishing points
    line1 = np.cross(v1, v2)
    line2 = np.cross(v3, v4)
    
    numerator = np.dot(line1.T, np.dot(omega_inv, line2))
    denom = np.sqrt(np.dot(line1.T, np.dot(omega_inv, line1))) * np.sqrt(np.dot(line2.T, np.dot(omega_inv, line2)))

    # Compute the angle between the two planes
    cos_theta = numerator/denom
    theta = np.arccos(cos_theta)
    return np.degrees(theta)
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE

    # first image
    d1i = np.array([np.dot(np.linalg.inv(K), np.array([v[0], v[1], 1.0]).T) / np.linalg.norm(np.dot(np.linalg.inv(K), np.array([v[0], v[1], 1.0]).T)) for v in vanishing_points1])
    # second image
    d2i = np.array([np.dot(np.linalg.inv(K), np.array([v[0], v[1], 1.0]).T) / np.linalg.norm(np.dot(np.linalg.inv(K), np.array([v[0], v[1], 1.0]).T)) for v in vanishing_points2])
    # the directional vectors in image 1 and image 2 are related by a rotation, R i.e. [d2i = R.d1i] => [R = d2i.d1i_inverse]
    R = np.dot(d2i.T, np.linalg.inv(d1i.T))
    return R

    # END YOUR CODE HERE

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print('Vanishing Points')
    print(floor_vanishing1)
    print(floor_vanishing2)
    print(box_vanishing1)
    print(box_vanishing2)
    print()
    print("Angle between floor and box:", angle)

    # # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
