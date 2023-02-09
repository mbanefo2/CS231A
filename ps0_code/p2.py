# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import crop, img_as_float64

def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.
    # Hint: use io.imread to read in the files

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = io.imread('image1.jpg')
    img2 = io.imread('image2.jpg')
    # END YOUR CODE HERE

    return img1, img2

def normalize_img(img):
    min_val = np.min(img)
    img = np.subtract(img, min_val)
    max_val = np.max(img)
    img = 255/max_val * img
    
    return img / np.max(img)

def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = img_as_float64(img1)
    img2 = img_as_float64(img2)
    
    img1 = normalize_img(img1)
    img2 = normalize_img(img2)

    # END YOUR CODE HERE
    
    return img1, img2
    
def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None
    
    # BEGIN YOUR CODE HERE
    sumImage = np.add(img1, img2)
    sumImage = normalize_img(sumImage)

    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(sumImage)
    io.imsave('p2_part_c.png', sumImage)
    return sumImage

def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.
    newImage1 = None

    # BEGIN YOUR CODE HERE
    img1_shape = img1.shape
    img2_shape = img2.shape
    
    left_half_image1 = img1[:, :img1_shape[1]//2]
    right_half_image2 = img2[:, img2_shape[1]//2:]
    
    newImage1 = np.concatenate((left_half_image1, right_half_image2), axis=1)

    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage1)
    io.imsave('p2_part_d.png', newImage1)
    return newImage1

def part_e(img1, img2):    
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.
    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros(img1.shape)
    
    for i in range(0, img1.shape[0]):
        if (i % 2) == 0:
            newImage2[i,:,:] = img2[i,:,:]
        else:
            newImage2[i,:,:] = img1[i,:,:]
    
    io.imsave('p2_part_e.png', newImage2)

    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage2)
    return newImage2

def part_f(img1, img2):     
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.
    newImage3 = None

    # BEGIN YOUR CODE HERE
    newImage3 = np.zeros(img1.shape)
                         
    # Assign the even rows of image1 to the new image
    newImage3[::2, :, :] = img2[::2, :, :]

    # Assign the odd rows of image2 to the new image
    newImage3[1::2, :, :] = img1[1::2, :, :]

    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage3)
    io.imsave('p2_part_f.png', newImage3)
    
    return newImage3

def part_g(img):         
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.
    # Hint: use np.dot and the standard formula for converting RGB to grey
    # greyscale = R*0.299 + G*0.587 + B*0.114
    grayImage = None

    # BEGIN YOUR CODE HERE
    greyscale_matrix = np.array([[0.299, 0.587, 0.114]])
    grayImage = np.dot(img, greyscale_matrix.T)

    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(grayImage)
    io.imsave('p2_part_g.png', grayImage)
    return grayImage

if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    grayImage = part_g(newImage3)