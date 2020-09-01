import cv2 as cv
import numpy as np
import os
from imgaug import augmenters as iaa
import random
import math
import csv
from PIL import Image, ImageEnhance

"""
Preprocessing methods:
- grayscaling
- blurring
- cropping
- weather augmentation, e.g. snow, fog, clouds
- rotating
- adjusting the brightness, e.g. lighter or darker
Parameters:
- amount of generated pairs
- amount of snow
- strength of blurring filter
- cropping size
"""


def write_to_metadata(metadata, method):
    if method in metadata:
        metadata[method] += 1
    else:
        metadata[method] = 1

    return metadata


def grayscaling(image1, image2, name, metadata):
    image_gray_1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image_gray_2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    name_1 = name + '_A_g.jpg'
    name_2 = name + '_B_g.jpg'
    cv.imwrite(name_1, image_gray_1)
    cv.imwrite(name_2, image_gray_2)
    metadata = write_to_metadata(metadata, "grayscaling")

    return metadata


def blurring(image1, image2, name, metadata):
    random_range = (0.5,1.5)
    blur = iaa.GaussianBlur(sigma=random_range)
    image_blur_1 = blur.augment_image(image1)
    image_blur_2 = blur.augment_image(image2)
    name_1 = name + '_A_b.jpg'
    name_2 = name + '_B_b.jpg'
    cv.imwrite(name_1, image_blur_1)
    cv.imwrite(name_2, image_blur_2)
    metadata = write_to_metadata(metadata, "blurring")

    return metadata


def cropping(image1, image2, name, metadata):
    height, width, _ = image1.shape
    crop_1 = iaa.CropToFixedSize(int(width*0.8), int(height*0.8), position="left-top")
    crop_2 = iaa.CropToFixedSize(int(width*0.8), int(height*0.8), position="right-bottom")
    image_crop_1 = cv.resize(crop_1.augment_image(image1), (width, height))
    image_crop_2 = cv.resize(crop_2.augment_image(image2), (width, height))
    name_1 = name + '_A_p.jpg'
    name_2 = name + '_B_p.jpg'
    cv.imwrite(name_1, image_crop_1)
    cv.imwrite(name_2, image_crop_2)
    metadata = write_to_metadata(metadata, "cropping")

    return metadata


# https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
def add_snow(image):
    image_HLS = cv.cvtColor(image,cv.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    brightness_coefficient = 2.5
    snow_point=120 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv.cvtColor(image_HLS,cv.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB


# use the snowing preprocessing method just for certain areas by choosing just selected images (without randomness)
def snowing(image1, image2, name, metadata):
    image_snow_1 = add_snow(image1)
    image_snow_2 = add_snow(image2)
    name_1 = name + '_A_s.jpg'
    name_2 = name + '_B_s.jpg'
    cv.imwrite(name_1, image_snow_1)
    cv.imwrite(name_2, image_snow_2)
    metadata = write_to_metadata(metadata, "snowing")

    return metadata


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height, around its centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


# https://code.i-harness.com/en/q/feddf6
def rotating(image1, image2, name, metadata):
    degree = random.randint(0, 360)
    height, width, _ = image1.shape
    image_rotated_1 = rotate_image(image1, degree)
    image_rotated_2 = rotate_image(image2, -degree)
    image_rotated_cropped_1 = crop_around_center(image_rotated_1, *largest_rotated_rect(width, height, math.radians(degree)))
    image_rotated_cropped_2 = crop_around_center(image_rotated_2, *largest_rotated_rect(width, height, math.radians(-degree)))
    image_rotated_final_1 = cv.resize(image_rotated_cropped_1, (width, height))
    image_rotated_final_2 = cv.resize(image_rotated_cropped_2, (width, height))
    name_1 = name + '_A_r.jpg'
    name_2 = name + '_B_r.jpg'
    cv.imwrite(name_1, image_rotated_final_1)
    cv.imwrite(name_2, image_rotated_final_2)
    metadata = write_to_metadata(metadata, "rotating")

    return metadata


def brightening(image1, image2, name, metadata):
    # convert cv image to pil image
    pil_image1 = Image.fromarray(image1)
    pil_image2 = Image.fromarray(image2)
    # enhance pil image by changing the brightness level
    enhancer1 = ImageEnhance.Brightness(pil_image1)
    enhancer2 = ImageEnhance.Brightness(pil_image2)
    # original = 1.0; brighter > 1.0
    brightness_level = 1.8
    brighter_image1 = enhancer1.enhance(brightness_level)
    brighter_image2 = enhancer2.enhance(brightness_level)
    # converting pil image back to cv image
    brighter_image_cv_1 = np.array(brighter_image1)
    brighter_image_cv_2 = np.array(brighter_image2)
    name_1 = name + '_A_l.jpg'
    name_2 = name + '_B_l.jpg'
    cv.imwrite(name_1, brighter_image_cv_1)
    cv.imwrite(name_2, brighter_image_cv_2)
    metadata = write_to_metadata(metadata, "brightening")

    return metadata


def darkening(image1, image2, name, metadata):
    # convert cv image to pil image
    pil_image1 = Image.fromarray(image1)
    pil_image2 = Image.fromarray(image2)
    # enhance pil image by changing the brightness level
    enhancer1 = ImageEnhance.Brightness(pil_image1)
    enhancer2 = ImageEnhance.Brightness(pil_image2)
    # original = 1.0; darker < 1.0
    brightness_level = 0.5
    darker_image1 = enhancer1.enhance(brightness_level)
    darker_image2 = enhancer2.enhance(brightness_level)
    # converting pil image back to cv image
    darker_image_cv_1 = np.array(darker_image1)
    darker_image_cv_2 = np.array(darker_image2)
    name_1 = name + '_A_d.jpg'
    name_2 = name + '_B_d.jpg'
    cv.imwrite(name_1, darker_image_cv_1)
    cv.imwrite(name_2, darker_image_cv_2)
    metadata = write_to_metadata(metadata, "darkening")

    return metadata


def adding_fog(image1, image2, name, metadata):
    fog = iaa.Fog()
    image_fog1 = fog.augment_image(image1)
    image_fog2 = fog.augment_image(image2)
    name_1 = name + '_A_f.jpg'
    name_2 = name + '_B_f.jpg'
    cv.imwrite(name_1, image_fog1)
    cv.imwrite(name_2, image_fog2)
    metadata = write_to_metadata(metadata, "fog")

    return metadata


def adding_clouds(image1, image2, name, metadata):
    cloud = iaa.CloudLayer(
        intensity_mean=(196, 255), intensity_freq_exponent=(-1.5, -2.0), intensity_coarse_scale=10,
        alpha_min=0, alpha_multiplier=(0.25, 0.75), alpha_size_px_max=(2, 8), alpha_freq_exponent=(-2.5, -2.0),
        sparsity=(0.8, 1.0), density_multiplier=(2.0, 2.5))
    image_cloud1 = cloud.augment_image(image1)
    image_cloud2 = cloud.augment_image(image2)
    name_1 = name + '_A_c.jpg'
    name_2 = name + '_B_c.jpg'
    cv.imwrite(name_1, image_cloud1)
    cv.imwrite(name_2, image_cloud2)
    metadata = write_to_metadata(metadata, "clouds")

    return metadata


# load original images as pairs
def name_list(n, path = ''):
    # range_min describes the number which the name of the first image pair starts
    # range_max is the number from the name of the last image pair

    range_min = 1
    range_max = n+1
    
    if path == '':
        A = ['%04d_A.jpg'%(i) for i in range(range_min,range_max)]
        B = ['%04d_B.jpg'%(i) for i in range(range_min,range_max)]
    else:
        A = [os.path.join(path, '%04d_A.jpg'%(i)) for i in range(range_min,range_max)]
        B = [os.path.join(path, '%04d_B.jpg'%(i)) for i in range(range_min,range_max)]
    return A, B


def load_image_pairs(path):
    l = sorted(os.listdir(path))
    l = [x for x in l if 'jpg' in x.lower()]
    assert len(l) % 2 == 0
    A_list, B_list = name_list( len(l) // 2, path=path)

    A = np.array([cv.imread(fname) for fname in A_list])
    B = np.array([cv.imread(fname) for fname in B_list])

    return A, B


def run(new, A, B, preprocessed_image_path, metadata, methods):
    # randomly choose pairs
    size_dataset = len(A)

    if new <= size_dataset:
        randomness = False
    else:
        randomness = True

    random_indices = list()
    for i in range(new):
        if randomness:
            random_indices.append(random.randint(0,size_dataset-1))
        else:
            random_indices.append(i)
    count = 1
    offset = 0
    # randomly choose preprocessing methods for the chosen pairs
    for i in random_indices:
        image1 = A[i]
        image2 = B[i]
        number = "%04d" % (i + 1 + offset)
        name = preprocessed_image_path + number
        metadata = random.choice(methods)(image1, image2, name, metadata)
        print("Pair: #", count)
        count += 1

    return metadata
    
def main():
    image_path = "filepath/image"
    
    # always use an empty directory to store the generated images
    preprocessed_image_path = "filepath/augmented_image"
    
    # write generated dictionary with metadata to csv file
    csv_metadata = "filepath/metadata.csv"
    
    # write list of filenames to csv file
    csv_filename = "filepath/list.csv"

    # list of available preprocessing methods
    methods = [grayscaling, blurring, cropping, snowing, rotating, adding_fog, adding_clouds, brightening, darkening]

    # read original images as pairs
    A, B = load_image_pairs(image_path)

    # choose the amount of generated pairs with maximum amount = amount of available originals * amount of available preprocessing methods
    pairs = 50
    images = 2 * pairs
    missing = images
    a = list()
    metadata = dict()

    # generate the images until the desired amount is achieved + the two csv files
    # (doubled randomly chosen images get overwritten! thats why we need several rounds)
    while len(a) < images:
        metadata = run(int(missing / 2), A, B, preprocessed_image_path, metadata, methods)
        a = os.listdir(preprocessed_image_path)
        missing = images - len(a)
        print("Created images:", len(a))
        print("Missing images:", missing)

    with open(csv_metadata, 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        for key, value in metadata.items():
            row = [key, value]
            writer.writerow(row)

    csvFile.close()

    list_of_names = list()
    # write list of filenames to csv file
    for file in a:
        if file.endswith(".jpg"):
            list_of_names.append(file)

    with open(csv_filename, 'w') as csvFile:
        for i in range(1, len(list_of_names) + 1, 2):
            row = [list_of_names[i - 1][-12:-4], list_of_names[i][-12:-4]]
            writer = csv.writer(csvFile, lineterminator='\n')
            writer.writerow(row)

    csvFile.close()
    
if __name__ == '__main__':
    main()
