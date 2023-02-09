# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    # Hint: reshape your values such that you have PM=p,
    # and use np.linalg.lstsq or np.linalg.pinv to solve for M.
    # See https://apimirror.com/numpy~1.11/generated/numpy.linalg.pinv
    #
    # Our solution has shapes for M=(8,), P=(48,8), and p=(48,)
    # Alternatively, you can set things up such that M=(4,2), P=(24,4), and p=(24,2)
    # Lastly, reshape and add the (0,0,0,1) row to M to have it be (3,4)

    # BEGIN YOUR CODE HERE
    camera_matrix = None
    P, small_p = generate_P(real_XY, front_image, back_image)
    M,_,_,_ = np.linalg.lstsq(P, small_p, rcond=None)
    M = M.reshape((2,4))

    last_row = np.array([0,0,0,1])
    camera_matrix = np.vstack([M, last_row])

    return camera_matrix

    # END YOUR CODE HERE

def generate_P(real_XY, front_image, back_image):
    '''
    Helper function to generate P and small_p (image coordinate vector)
    '''
    P = None

    #  Add Z coordinates and convert to homogeneous
    zero_vector = np.zeros((1,12))
    ones_vector = np.ones((1,12))
    back_array = np.full((1,12), 150)
    
    front_image_new = np.vstack([real_XY.T, zero_vector, ones_vector])
    back_image_new = np.vstack([real_XY.T, back_array, ones_vector])
    
    total_image = np.concatenate((front_image_new,back_image_new), axis=1 ).T
    original_total_image = np.concatenate((front_image,back_image), axis=0 )
    
    # Create array shapes for P and small_p
    P = np.zeros((48,8))
    small_p = np.zeros((48))
    
    # Calculate P
    for j in range(total_image.shape[0]):
        P[(j*2)] = [total_image[j][0], total_image[j][1], total_image[j][2], total_image[j][3], 0, 0, 0 ,0]
        P[(j*2) + 1] = [0, 0, 0 ,0, total_image[j][0], total_image[j][1], total_image[j][2], total_image[j][3]]
    
    # Calculate small p
    for j in range(original_total_image.shape[0]):
        small_p[(j*2)] = original_total_image[j][0]
        small_p[(j*2)+1] = original_total_image[j][1]
    
    return P, small_p
'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # BEGIN YOUR CODE HERE
    RMS = None
    
    # Get P and small_p (corner coordinates)
    P, small_p = generate_P(real_XY, front_image, back_image)
    
    # Remove the last row of 0001 and reshape to 24,2 for x,y coordinates
    updated_cam_mat = np.delete(camera_matrix,2,0)
    result = np.dot(P, updated_cam_mat.reshape(8))
    result = result.reshape((24,2))
    small_p = small_p.reshape((24,2))
    
    # Calculate RMS Error
    RMS = np.sqrt(np.mean(np.sum((result - small_p)**2, axis=1)))
    
    return RMS
    # END YOUR CODE HERE

if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
