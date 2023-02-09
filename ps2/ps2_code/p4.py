import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    u, s, vt = np.linalg.svd(E)
    W = np.zeros((3,3))
    Z = np.zeros((3,3))
    
    W[0,1] = -1
    W[1,0] = 1
    W[2,2] = 1
   
    Z[0,1] = 1
    Z[1,0] = -1
    
    Q1 = np.dot(np.dot(u, W), vt)
    Q2 = np.dot(np.dot(u, W.T), vt)
    
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2
    
    T1 = np.array(u[:,2])
    T2 = np.array(-u[:,2])
    
    RT = np.zeros((4,3,4), dtype=np.float32)
    for i, (R, T) in enumerate(zip((R1, R1, R2, R2), (T1, T2, T1, T2))):
        RT[i][:,:3] = R
        RT[i][:,3] = T 

    # return initial RT
    return RT
'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):

    # Compute matrix A
    A = []
    for i, M in enumerate(camera_matrices):
        A.append(image_points[i][0]*M[2] - M[0])
        A.append(image_points[i][1]*M[2] - M[1])
    A = np.array(A, dtype=np.float64)
    
    u ,s ,vt = np.linalg.svd(A)
    
    # Select 3D point as last row of vt
    three_d = vt[-1]
    
    # Homogeneous to Eucledian
    three_d = (three_d / three_d[-1])[:3]
    three_d = np.array(three_d, dtype=np.float64)

    return three_d

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
 
    error = []
    for i, M in enumerate(camera_matrices):
        # convert 3D to homogenous
        h_3d = np.append(point_3d, 1.0)
        y = np.dot(M, h_3d)
        new_im_point = y / y[2]
        new_im_point = new_im_point[:2]
        error.extend(new_im_point - image_points[i])
        
    return np.array(error, dtype=np.float64)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    h_3d = np.append(point_3d, 1.0)
    
    J = []
    for M in camera_matrices:
        m1_3d_hat = np.dot(M[0,:], h_3d.T)
        m2_3d_hat = np.dot(M[1,:], h_3d.T)
        m3_3d_hat = np.dot(M[2,:], h_3d.T)
        
        J.append(
                [
                    ((M[0,0]*m3_3d_hat - M[2,0]*m1_3d_hat) / (m3_3d_hat ** 2)),
                    ((M[0,1]*m3_3d_hat - M[2,1]*m1_3d_hat) / (m3_3d_hat ** 2)),
                    ((M[0,2]*m3_3d_hat - M[2,2]*m1_3d_hat) / (m3_3d_hat ** 2))
                ]
            )
        J.append(
                [
                    ((M[1,0]*m3_3d_hat - M[2,0]*m2_3d_hat) / (m3_3d_hat ** 2)),
                    ((M[1,1]*m3_3d_hat - M[2,1]*m2_3d_hat) / (m3_3d_hat ** 2)),
                    ((M[1,2]*m3_3d_hat - M[2,2]*m2_3d_hat) / (m3_3d_hat ** 2))
                ]
            )
    J = np.array(J, dtype=np.float32)

    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    points_3d = linear_estimate_3d_point(image_points, camera_matrices)
    
    for i in range(10):
        # Update the Jacobian and error with each iteration
        J = jacobian(points_3d, camera_matrices)
        e = reprojection_error(points_3d, image_points, camera_matrices)
        points_3d = points_3d - np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), e)
    
    return points_3d

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # Calculate the initial RT from the Essential Matrix (E)
    RT_init = estimate_initial_RT(E)
    # Initialize an array to store the 3D points for each RT
    points_3d_rt = np.zeros((4, image_points.shape[0], 2, 3), dtype=np.float32)

    for i, RT in enumerate(RT_init):
        for j, image_point in enumerate(image_points):
            # Construct the camera matrix for camera 1 and 2 and stack
            cam_mtx_1 = np.hstack((K, np.zeros((3,1), dtype=np.float32)))
            cam_mtx_2 = np.dot(K, RT)
            camera_matrices = np.array([cam_mtx_1, cam_mtx_2], dtype=np.float32)

            # Calculate the non-linear estimate of the 3D point in homogenous
            point_3d = nonlinear_estimate_3d_point(image_point, camera_matrices)
            point_3d_homogeneous = np.append(point_3d, 1.0)
            
            # Store the 3D point in the array for camera 1 and 2
            points_3d_rt[i,j,0] = point_3d
            points_3d_rt[i,j,1] = np.dot(RT, point_3d_homogeneous)
    
    correct_RT = None
    max_count = 0
    
    for i in range(RT_init.shape[0]):
        # Count the number of points that fall in front of both cameras
        count = np.sum(points_3d_rt[i,:,:,2] > 0.)

        if count > max_count:
            max_count = count
            correct_RT = RT_init[i]

    return correct_RT

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
