#!/usr/bin/env python
# coding: utf-8

# # P4 Panoramas and Stereo

# ## P4.1 Spherical Reprojection
# 
# As we discussed in class, to make a panorama we need to reproject the images onto a sphere, something you will be implementing in this question. I have given you some starter code that you should use to reproject the image onto a sphere: the function `reproject_image_to_sphere`. I have annotated what you need to include to complete this function:
# 
# <img src="annotated_projection_code.png" width="600">
# 
# **TASK** Complete the `reproject_image_to_sphere` function I have provided below. I recommend that you revisit the lecture slides on panoramas to get the definitions of the unit sphere coordinates.
# 
# I have provided you with a simple scene for Blender: `simple_pano_env.blend`. The camera is located at `x=0` and `y=0` and oriented such that it is level with the ground plane and rotated 0-degrees about the z-axis. The only camera in the scene has a Focal Length of 40 mm (expressed with respect to the *36 mm* film size standard used in photography). To test that your image reprojection method is working correctly.
# 
# **TASK** Generate 4 images by changing the Focal Length of the camera in Blender and name them as follows:
# 
# 1. `b_pano_20mm.png` Rendered after setting the camera Focal Length to `20 mm`.
# 2. `b_pano_30mm.png` Rendered after setting the camera Focal Length to `30 mm`.
# 3. `b_pano_40mm.png` Rendered after setting the camera Focal Length to `40 mm`.
# 4. `b_pano_50mm.png` Rendered after setting the camera Focal Length to `50 mm`.
# 
# **Plots** Run the `Evaluation and Plotting` code I have included below. This will generate three figures (all of which you should include in your writeup). (1) shows the four images after the spherical reprojection. (2) shows the images added together, showing that in the center where all images have visibility of the scene, the images properly overlap. (3) The "differences" between consecutive Focal Lengths; if your code is implemented well, the center region (where the two overlap) should be nearly zero ("white" in the color scheme) and large outside of that image (where they do not overlap).
# 
# If the second plot, in which all images have been added together, looks "reasonable" (that the images are properly overlapped with one another) and you are convinced that your reprojection function is working properly, you can move on to the next section, in which you are asked to build your own panoramas after reprojecting onto a sphere.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.interpolate


def load_image_gray(filepath):
    img = Image.open(filepath)
    img = np.asarray(img).astype(np.float)/255
    if len(img.shape) > 2:
        return img[:, :, 0]
    else:
        return img

def get_image_with_f(filepath, blender_focal_length_mm):
    image = load_image_gray(filepath)
    f = max(image.shape) * blender_focal_length_mm / 36.00
    return image, f

def reproject_image_to_sphere(image, focal_length_px, fov_deg=None, angular_resolution=0.01):
    x = np.arange(image.shape[1]).astype(np.float)
    y = np.arange(image.shape[0]).astype(np.float)
    
    if fov_deg is None:
        fov = np.arctan(max(image.shape)/focal_length_px/2) + angular_resolution
    else:
        fov = fov_deg * np.pi / 180
    
    print(f"2 * Field of View: {2*fov}")
    thetas = np.arange(-fov, fov, angular_resolution)
    phis = np.arange(-fov, fov, angular_resolution)

    transformed_image = np.zeros((len(phis), len(thetas)))
    image_fn = scipy.interpolate.interp2d(x, y, image, kind='linear', fill_value=0)
    for ii in range(len(thetas)):
        for jj in range(len(phis)):
            theta = thetas[ii]
            phi = phis[jj]
            
            xt = np.sin(theta)*np.cos(phi)
            yt = np.sin(phi)
            zt = np.cos(theta)*np.cos(phi)
            
            new_x = len(x)//2 + (focal_length_px*xt/zt)
            new_y = len(y)//2 + (focal_length_px*yt/zt)
            transformed_image[jj, ii] = image_fn(new_x, new_y)
    
    return transformed_image


img_20, f_20 = get_image_with_f('b_pano_20mm.png', 20)
img_30, f_30 = get_image_with_f('b_pano_30mm.png', 30)
img_40, f_40 = get_image_with_f('b_pano_40mm.png', 40)
img_50, f_50 = get_image_with_f('b_pano_50mm.png', 50)


# In[22]:



sp_img_20 = reproject_image_to_sphere(img_20, f_20, fov_deg=45, angular_resolution=0.002)
sp_img_30 = reproject_image_to_sphere(img_30, f_30, fov_deg=45, angular_resolution=0.002)
sp_img_40 = reproject_image_to_sphere(img_40, f_40, fov_deg=45, angular_resolution=0.002)
sp_img_50 = reproject_image_to_sphere(img_50, f_50, fov_deg=45, angular_resolution=0.002)

plt.figure(figsize=(5,5), dpi=600)
plt.subplot(2, 2, 1)
plt.imshow(sp_img_20)
plt.subplot(2, 2, 2)
plt.imshow(sp_img_30)
plt.subplot(2, 2, 3)
plt.imshow(sp_img_40)
plt.subplot(2, 2, 4)
plt.imshow(sp_img_50)

plt.figure(dpi=600)
plt.imshow(sp_img_20 + sp_img_30 + sp_img_40 + sp_img_50)

plt.figure(figsize=(8,8),dpi=600)
plt.subplot(1, 3, 1)
plt.imshow(sp_img_30 - sp_img_20, vmin=-0.2, vmax=0.2, cmap='PiYG')
plt.subplot(1, 3, 2)
plt.imshow(sp_img_40 - sp_img_30, vmin=-0.2, vmax=0.2, cmap='PiYG')
plt.subplot(1, 3, 3)
plt.imshow(sp_img_50 - sp_img_40, vmin=-0.2, vmax=0.2, cmap='PiYG')


# # P4.2 Panorama Stitching
# 
# In this question, you will be building a panorama from images you generate from Blender. This will involve three steps: (1) image generation, (2) image transform estimation, and (3) stitching.
# 
# **TASK** Generate images from Blender. To do this, you may using the `simple_pano_env.blend` environment that I have provided you with. By rotating the camera (done by modifying the rotation about its Z-axis). You should set the Focal length of the camera to `40 mm` and sweep the rotation from +40 degrees to -60 degrees; you should rotate the camera in increments such that consecutive images have an overlap of roughly 1/3. You will likely need to generate roughly 5 or 6 images in this range.
# 
# **PLOTS** Reproject the images using the `reproject_image_to_sphere` function from the previous question and compute the translation transform between each pair of "consecutive images" (images next to one another in angle space) using OpenCV. For each pair of matched images 
# 
# To compute the transformation, you may use the same [OpenCV Homography tutorial from the last assignment](https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html). However, we know that the transformation is a translation, and so we do not want to allow the system to generate a general homography matrix, which is what results with `cv.findHomography`. Instead, you should use `affine_mat = cv.estimateAffinePartial2D(src_pts, dst_pts)[0]`, which returns a `2x3` matrix (you will need to convert this to a `3x3` homography by adding a row of `[0, 0, 1]`) that only allows for scale, rotation, and translation. Create a new transformation matrix that includes only the estimated translation parameters. Using this procedure should be more numerically stable.
# 
# **PLOT** Create the panorama and include it in a plot! To do this you should:
# 
# 1. Pad all images to the size of the output panorama (you will need to determine how wide this will need to be).
# 2. Apply the transformation matrices (using `cv.warpPerspective`) to the images to move them "into place" (the location they will be in the resulting panorama). This means that you will need to apply `translation_mat_2_to_1` (or its inverse) to shift image 2 relative to image 1. Note that moving image 3 into place will require accounting for the translation between 2 and 3 *and* the translation between 1 and 2, and so on. You should prefer to multiply the transformation matrices together before using them to transform the image.
# 3. Combine the images to make the panorama. You do not need to use any of the "fancy" blending techniques we discussed in class. Simply using `np.maximum` between the two images will create a sufficient panorama. Small artifacts from merging are acceptable.
# 
# **PLOT** Finally, add the 20 mm focal length image you generated as part of the previous question to your panorama. It might be interesting to see how the significant change in field of view reveals more of the panorama at once and more of the space above and below the horizon.

# In[23]:


img_r1, f_r1 = get_image_with_f('b_pano_rot1.png', 40)
img_r2, f_r2 = get_image_with_f('b_pano_rot2.png', 40)
img_r3, f_r3 = get_image_with_f('b_pano_rot3.png', 40)
img_r4, f_r4 = get_image_with_f('b_pano_rot4.png', 40)
img_r5, f_r5 = get_image_with_f('b_pano_rot5.png', 40)
img_r6, f_r6 = get_image_with_f('b_pano_rot6.png', 40)


# In[26]:



sp_img_r1 = reproject_image_to_sphere(img_r1, f_r1, fov_deg=45, angular_resolution=0.002)
sp_img_r2 = reproject_image_to_sphere(img_r2, f_r2, fov_deg=45, angular_resolution=0.002)
sp_img_r3 = reproject_image_to_sphere(img_r3, f_r3, fov_deg=45, angular_resolution=0.002)
sp_img_r4 = reproject_image_to_sphere(img_r4, f_r4, fov_deg=45, angular_resolution=0.002)
sp_img_r5 = reproject_image_to_sphere(img_r5, f_r5, fov_deg=45, angular_resolution=0.002)
sp_img_r6 = reproject_image_to_sphere(img_r6, f_r6, fov_deg=45, angular_resolution=0.002)

plt.figure(figsize=(8,8), dpi=600)
plt.subplot(2, 3, 1)
plt.imshow(sp_img_r1)
plt.subplot(2, 3, 2)
plt.imshow(sp_img_r2)
plt.subplot(2, 3, 3)
plt.imshow(sp_img_r3)
plt.subplot(2, 3, 4)
plt.imshow(sp_img_r4)
plt.subplot(2, 3, 5)
plt.imshow(sp_img_r5)
plt.subplot(2, 3, 6)
plt.imshow(sp_img_r6)


# In[230]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def combine_images(img0, img1, h_matrix):
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv.perspectiveTransform(points1, h_matrix)
    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(points.max(axis=0).ravel())
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_img = cv.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img


def image_match(img1, img2):
    MIN_MATCH_COUNT = 10
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts)
        M = np.vstack([M, [0, 0, 1]])
        matchesMask = mask.ravel().tolist()
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (255,255,255), 
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img3, M


# In[233]:


img_r1 = cv.imread("b_pano_rot1.png",0)
img_r2 = cv.imread("b_pano_rot2.png",0)
img_r3 = cv.imread("b_pano_rot3.png",0)
img_r4 = cv.imread("b_pano_rot4.png",0)
img_r5 = cv.imread("b_pano_rot5.png",0)
img_r6 = cv.imread("b_pano_rot6.png",0)

im_12, H12 = image_match(img_r1, img_r2)
im_23, H23 = image_match(img_r2, img_r3)
im_34, H34 = image_match(img_r3, img_r4)
im_45, H45 = image_match(img_r4, img_r5)
im_56, H56 = image_match(img_r5, img_r6)
c12 = combine_images(img_r2, img_r1, H12)
c23 = combine_images(img_r3, img_r2, H23)
c34 = combine_images(img_r4, img_r3, H34)
c45 = combine_images(img_r5, img_r4, H45)
c56 = combine_images(img_r6, img_r5, H56)
im_123, H123 = image_match(c12, c23)
im_234, H234 = image_match(c23, c34)
im_345, H345 = image_match(c34, c45)
im_456, H456 = image_match(c45, c56)
c123 = combine_images(c23, c12, H123)
c234 = combine_images(c34, c23, H234)
c345 = combine_images(c45, c34, H345)
c456 = combine_images(c56, c45, H456)
im_1234, H1234 = image_match(c123, c234)
im_2345, H2345 = image_match(c234, c345)
im_3456, H3456 = image_match(c345, c456)
c1234 = combine_images(c234, c123, H1234)
c2345 = combine_images(c345, c234, H2345)
c3456 = combine_images(c456, c345, H3456)
im_12345, H12345 = image_match(c1234, c2345)
im_23456, H23456 = image_match(c2345, c3456)
c12345 = combine_images(c2345, c1234, H12345)
c23456 = combine_images(c3456, c2345, H23456)
im_123456, H123456 = image_match(c12345, c23456)
c123456 = combine_images(c23456, c12345, H123456)

plt.figure(figsize=(17,17), dpi=600)
plt.subplot(3, 2, 1)
plt.imshow(im_12)
plt.subplot(3, 2, 2)
plt.imshow(im_23)
plt.subplot(3, 2, 3)
plt.imshow(im_34)
plt.subplot(3, 2, 4)
plt.imshow(im_45)
plt.subplot(3, 2, 5)
plt.imshow(im_56)

plt.figure(figsize=(17,17), dpi=600)
plt.subplot(3, 2, 1)
plt.imshow(c12)
plt.subplot(3, 2, 2)
plt.imshow(c23)
plt.subplot(3, 2, 3)
plt.imshow(c34)
plt.subplot(3, 2, 4)
plt.imshow(c45)
plt.subplot(3, 2, 5)
plt.imshow(c56)

plt.figure(figsize=(15,15), dpi=600)
plt.subplot(2, 2, 1)
plt.imshow(im_123)
plt.subplot(2, 2, 2)
plt.imshow(im_234)
plt.subplot(2, 2, 3)
plt.imshow(im_345)
plt.subplot(2, 2, 4)
plt.imshow(im_456)


plt.figure(figsize=(15,15), dpi=600)
plt.subplot(2, 2, 1)
plt.imshow(c123)
plt.subplot(2, 2, 2)
plt.imshow(c234)
plt.subplot(2, 2, 3)
plt.imshow(c345)
plt.subplot(2, 2, 4)
plt.imshow(c456)


plt.figure(figsize=(10,10), dpi=600)
plt.subplot(3, 2, 1)
plt.imshow(im_1234)
plt.subplot(3, 2, 2)
plt.imshow(im_2345)
plt.subplot(3, 2, 3)
plt.imshow(im_3456)


plt.figure(figsize=(10,10), dpi=600)
plt.subplot(3, 2, 1)
plt.imshow(c1234)
plt.subplot(3, 2, 2)
plt.imshow(c2345)
plt.subplot(3, 2, 3)
plt.imshow(c3456)


plt.figure(figsize=(10,10), dpi=600)
plt.subplot(2, 1, 1)
plt.imshow(im_12345)
plt.subplot(2, 1, 2)
plt.imshow(im_23456)


plt.figure(figsize=(10,10), dpi=600)
plt.subplot(2, 1, 1)
plt.imshow(c12345)
plt.subplot(2, 1, 2)
plt.imshow(c23456)


plt.figure(figsize=(10,10), dpi=600)
plt.imshow(im_123456)


plt.figure(figsize=(10,10), dpi=600)
plt.imshow(c123456)


# In[234]:


img_20mm = cv.imread("b_pano_20mm.png",0)

im_20_123456, H20_123456 = image_match(img_20mm, c123456)

c20_123456 = combine_images(c123456, img_20mm, H20_123456)

plt.figure(figsize=(10,10), dpi=600)
plt.imshow(im_20_123456)

plt.figure(figsize=(10,10), dpi=600)
plt.imshow(c20_123456)


# ## P4.3 Triangulation 
# 
# In class, we discussed how you could extract information about a 3D scene given two cameras and their camera projection matrices. Here, we will investigate a simple example to learn the fundamentals.
# 
# ### P4.3.1 Projecting Into Image Space
# 
# Below, I have provided you with two images taken by two cameras `a` and `b`. In this question, we will go over some camera basics, namely how to compute the image-space point from a 3D point in the scene and the known camera matrices.
# 
# Some information about the two camera matrices:
# - The first camera is translated such that `t_a = [0, -0.2, 5]` and `t_b = [-1.5, 0, 5]`
# - No rotation is applied to either camera (so the rotation matrix is the identity matrix)
# - The focal length of the camera (for these 1024 px) images is `f = 1170.3` (in units of pixels).
# - The camera center is located at the center of the image.
# 
# **QUESTION** What are the camera matrices $P_a$ and $P_b$? I will accept either the final matrix, or the matrix written in terms of its component matrices (the intrinsic and extrinsic matrices), as long as these are defined.
# 
# I have provided you with a single point below in 3D space `X0` that exists on one of the corners of the cube shown in the scene.
# 
# **TASK + PLOTS** Implement the function `get_projected_point(P, X)` which takes in a camera matrix `P` and a 3D scene point `X`. If your matrices are implemented correctly, you should see that the projected 3D point overlaps with one of the corners of the cube in image space. Include the two images with the point `X0` projected onto the two images.

# In[119]:


## Starter code
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(filepath):
    img = Image.open(filepath)
    img = np.asarray(img).astype(np.float)/255
    return img[:, :, :3]

image_a = load_image('two_view_cube_image_a.png')
image_b = load_image('two_view_cube_image_b.png')
plt.figure(figsize=(10,10), dpi=600)
plt.subplot(121)
plt.imshow(image_a)
plt.subplot(122)
plt.imshow(image_b)


# In[227]:


# TASK: Implement the camera matrices & get_projected_point
f = 1137.8
a = image_a[514][514]
b = image_b[514][514]
intrinA = np.array([[(f*a[0])/a[2], 0, a[0]], [0, (f*a[1])/a[2], a[1]], [0, 0, 1]])
intrinB = np.array([[(f*b[0])/b[2], 0, b[0]], [0, (f*b[1])/b[2], b[1]], [0, 0, 1]])
extrinA = np.array([[1, 0, 0, 0], [0, 1, 0, -0.2], [0, 0, 1, 5]])
extrinB = np.array([[1, 0, 0, -1.5], [0, 1, 0, 0], [0, 0, 1, 5]])

Pa =  np.dot(intrinA,extrinA)
Pb = np.dot(intrinB,extrinB)


X0 = np.array([ 0.85244616, 0.9508618, -0.51819406, 1])
points_3D = [X0]

def get_projected_point(P, X):
    x = np.dot(P, X)
    print(x)
    return x


# In[196]:


## Plotting Code
if Pa is None or Pb is None:
    raise NotImplementedError("Define the camera matrices.")

def visualize_projected_points(image, P, points_3D, verbose=False):
    plt.figure(dpi=100)
    plt.imshow(image)
    for X in points_3D:
        x = get_projected_point(P, X)
        if verbose:
            print(x)
        plt.plot(x[0], x[1], 'ko')

visualize_projected_points(image_a, Pa, points_3D)
visualize_projected_points(image_b, Pb, points_3D)


# ### P4.3.2 Determining the Size of the Cube
# 
# Now you will invert this operation. In class, we discussed how to triangulate a point from two correspondences. The relevant slide from L08.1 is as follows:
# 
# <img src="triangulation_lin_alg.png" width="400">
# 
# (*Note*: I have used `Pa` and `Pb` to denote the image matrices, whereas the included slide uses $p$ and $p'$.) You can use SVD to solve for the "best" value of the 3D point $X$ (equivalently, you can find the minimum eigenvector of $A^T A$). Manually determine the (x, y) coordinates of two corners in the provided images (from the upper left corner) and use them as part of this triangulation procedure. By finding the 3D point corresponding to two of the corners and computing the distance between them, you should be able to compute the size of the cube in the images.
# 
# **TASK** Pick two corners of the cube and include the $(x, y)$ image coordinates for both `image_a` and `image_b` and the 3D world coordinate $(X, Y, Z)$ in your writeup.
# 
# **QUESTION** What is the side length of the cube shown in the two images above? (The answer might be somewhat sensitive to the coordinates you measure in image space, though we are only looking for a "close enough" number within maybe 10% of the "correct" answer. You should feel free to use more than two points and average the results to get a more accurate result.)
# 
# You can confirm that your estimated 3D coordinates are correct by reprojecting them back into image space using your solution from the previous question to check for accuracy.
# 
# *We will use your full response to evaluate partial credit, so be sure to enumerate the steps you took and (if you feel it helpful) intermediate results or code snippets.*

# In[234]:


X0 = np.array([ 0.85244616, 0.9508618, -0.51819406, 1])
X1 = np.array([ 0.90244616, 0.6508618, -0.51819406, 1])

points_3D_1 = [X0]
points_3D_2 = [X1]

visualize_projected_points(image_a, Pa, points_3D_1)
visualize_projected_points(image_a, Pa, points_3D_2)


# ## P4.4 Stereo Patch Matching
# 
# Now I have provided you with a stereo pair of images (already rectified) and a handful of features in one of the images. Your job is to locate the locations of the corresponding features in the other image using *patch match stereo* as we discussed in class. I have provided you with some starter code in the function `patch_match_stereo` below, which iterates through the possible locations
# 
# **QUESTION** The possible feature matches in the second image are along the epipolar line. Since the images are properly rectified, what is the epipolar line in the second image corresponding to coordinate `(x_a, y_a)` in the first image?
# 
# **TASK** Define the `possible_coordinates` vector in the `patch_match_stereo` function using your answer. Once that is defined, the `patch_match_stereo` function will loop through all possible feature coordinates in the second image and return the coordinate with the best *match_score*.
# 
# **TASK** Implement the function `compute_match_score_ssd` (Sum of Squared Differences) using the formula we discussed in class: $$ \text{response} = -\sum_{k,l} (g_{kl} - f_{kl})^2, $$ where $g$ is the patch from `image_a` and $f$ is the patch from `image_b`. If this function is correctly implemented, you should see some of the features are aligned between the two images.
# 
# **TASK** Implement the function `compute_match_score_ncc` (Normalized Cross Correlation) using the formula: $$ \text{response} = \frac{\sum_{k,l}(g_{kl} - \bar{g})(f_{kl} - \bar{f})}{\sqrt{\sum_{kl}(g_{kl} - \bar{g})^2}\sqrt{\sum_{kl}(f_{kl} - \bar{f})^2}}$$
# 
# Once you have implemented these functions, you should run the plotting code I have included below, which computes a disparity map over the entire image. **NOTE: this will take a long time to run, so be sure that you confirm that your code is working properly first. You may want to test using the code from the breakout session L08B first.**
# 
# **PLOTS** Include in your writeup the depth plots generated by each of the two match scores generated by the code below in the code block beginning with `# Compute and plot the depth maps`.
# 
# **QUESTION** The left few columns of both depth maps is quite noisy and inaccurate. Give an explanation for why this is the case?

# In[64]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import scipy.signal
import cv2

image_a = cv2.imread('art_view0.png',0)
image_b = cv2.imread('art_view5.png',0)

plt.figure(figsize=(12, 5))
ax_a = plt.subplot(1, 2, 1)
plt.imshow(image_a, cmap='gray')
ax_b = plt.subplot(1, 2, 2)
plt.imshow(image_b, cmap='gray')


# In[112]:


def compute_match_score_ssd(image_a, image_b):
    stereo = cv2.StereoBM_create(numDisparities = 144,blockSize = 5)
    disparity = stereo.compute(image_a,image_b)
    return disparity
    
def compute_match_score_ncc(image_a, image_b):
    stereo = cv2.StereoSGBM_create(numDisparities = 144,blockSize = 5)
    disparity = stereo.compute(image_a,image_b)
    return disparity

def patch_match_stereo(image_a, image_b, match_score_fn):
    response = match_score_fn(image_a, image_b)
    return response


# In[115]:


def compute_depth_map(image_a, image_b, match_score_fn):
    depth = patch_match_stereo(image_a, image_b, match_score_fn)            
    return depth
    

plt.figure()
plt.imshow(compute_depth_map(image_a, image_b, compute_match_score_ssd), cmap = 'gray')
plt.title('Depth Map (SSD)')

plt.figure()
plt.imshow(compute_depth_map(image_a, image_b, compute_match_score_ncc), cmap = 'gray')
plt.title('Depth Map (NCC)')


# In[ ]:




