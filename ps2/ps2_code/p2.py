import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    u, s, v = np.linalg.svd(F.T)
    e = v[-1,:] / v[-1,-1]
    return e

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    T = compute_T(im)
    Te = np.dot(T, e)
    R = compute_R(Te)
    Re = np.dot(R,Te)
    G = compute_G(Re[0])
    H = np.dot(np.dot(np.dot(np.linalg.inv(T), G), R), T)

    return H
    
def compute_T(im):
    T = np.zeros((3,3))
    width = im.shape[1]
    height = im.shape[0]
    T[0,:] = [1, 0, -width/2]
    T[1,:] = [0, 1, -height/2]
    T[2,:] = [0,0,1]
    return T

def compute_R(e):
    R = np.zeros((3,3))
    
    alpha = 1 if e[0] >= 0 else -1

    R[0,:] = [alpha * e[0]/np.sqrt(e[0]**2 + e[1]**2), alpha * e[1]/np.sqrt(e[0]**2 + e[1]**2), 0]
    R[1,:] = [-alpha * e[1]/np.sqrt(e[0]**2 + e[1]**2), alpha * e[0]/np.sqrt(e[0]**2 + e[1]**2), 0]
    R[2,:] = [0,0,1]
    
    return R

def compute_G(f):
    G = np.zeros((3,3)) 
    G[0,:] = [1,0,0]
    G[1,:] = [0,1,0]
    G[2,:] = [-1/f, 0, 1]
    
    return G

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
''' 
def compute_ha(points1, points2, H2, M):
    W, b = construct_w_and_b(points1, points2, H2, M)
    a, residuals, rank, s = np.linalg.lstsq(W, b, rcond=None)
    
    Ha = np.zeros((3,3))
    Ha[0,:] = [a[0], a[1], a[2]]
    Ha[1,:] = [0,1,0]
    Ha[2,:] = [0,0,1]

    return Ha
    
def construct_w_and_b(points1, points2, H2, M):
    points1_hat = H2.dot(M.dot(points1.T)).T
    points2_hat = H2.dot(points2.T).T

    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]
    
    return W, b

def compute_matching_homographies(e2, F, im2, points1, points2):
    # calculate H2
    H2 = compute_H(e2, im2)

    # calculate H1
    # Cross product matrix
    e_cross = np.zeros((3,3))
    e_cross[0,:] = [0, -e2[2], e2[1]]
    e_cross[1,:] = [e2[2], 0, -e2[0]]
    e_cross[2,:] = [-e2[1], e2[0], 0]
    
    vt = np.array([1.0,1.0,1.0])
    M = np.dot(e_cross,F) + np.outer(e2,vt)
    
    Ha = compute_ha(points1, points2, H2, M)
    H1 = np.dot(np.dot(Ha, H2), M)

    return H1, H2


if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
