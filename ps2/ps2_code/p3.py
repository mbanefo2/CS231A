import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    points_im1_norm = normalize_points(points_im1)
    points_im2_norm = normalize_points(points_im2)
    
    # Construct matrix D
    D = np.zeros((4,points_im1.shape[0]))
    D[:2,:] = points_im1_norm[:,:2].T
    D[-2:,:] = points_im2_norm[:,:2].T
    
    # SVD of D
    u, s, vt = np.linalg.svd(D, full_matrices=True)
    
    motion = u[:,:3]
    s_mat = np.diag(s[:3])
    structure = np.dot(s_mat, vt[:3,:])

    # return motion and structure
    return structure, motion

def normalize_points(points):
    points_centroid = np.array((np.mean(points[:,0]), np.mean(points[:,1])), dtype=np.float32) 
    norm_points = points[:,:2] - points_centroid
    return norm_points

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
