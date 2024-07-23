import cv2
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from cv2 import dnn_superres




def mutual_information_registration(fixed_image_path, moving_image_path):

    # Read the fixed (optical) and moving (thermal) images
    fixed_image_cv = cv2.imread(fixed_image_path, cv2.IMREAD_UNCHANGED)
    moving_image_cv = cv2.imread(moving_image_path, cv2.IMREAD_UNCHANGED)

    # Resize the thermal image to the same size as the optical image
    if fixed_image_cv.shape[0] > 1000 or fixed_image_cv.shape[1] > 1000:
        scale_percent = 50  # Reduce size by 50%
        width = int(fixed_image_cv.shape[1] * scale_percent / 100)
        height = int(fixed_image_cv.shape[0] * scale_percent / 100)
        dim = (width, height)
        fixed_image_cv_resized = cv2.resize(fixed_image_cv, dim)
        moving_image_cv_resized = cv2.resize(moving_image_cv, dim)
    else:
        moving_image_cv_resized = cv2.resize(moving_image_cv, (fixed_image_cv.shape[1], fixed_image_cv.shape[0]))
        fixed_image_cv_resized = fixed_image_cv

    # Convert the resized images to SimpleITK format
    fixed_image_resized = sitk.GetImageFromArray(cv2.cvtColor(fixed_image_cv_resized, cv2.COLOR_BGR2GRAY).astype(np.float32))
    moving_image_resized = sitk.GetImageFromArray(moving_image_cv_resized.astype(np.float32))

    # Ensure the types are the same
    if fixed_image_resized.GetPixelID() != moving_image_resized.GetPixelID():
        moving_image_resized = sitk.Cast(moving_image_resized, fixed_image_resized.GetPixelID())

    # Initialize the transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_resized,
        moving_image_resized,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the image registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Use Mattes Mutual Information as the metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Set the interpolator to linear
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Use gradient descent optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100
    )

    # Set the optimizer scales from physical shift
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Set the initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image_resized, sitk.sitkFloat32),
        sitk.Cast(moving_image_resized, sitk.sitkFloat32)
    )

    # Print the final metric value and the optimizer's stopping condition
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print("Optimizer's stopping condition, {0}".format(registration_method.GetOptimizerStopConditionDescription()))

    # Resample the thermal image to align with the fixed image using the final transform
    moving_resampled = sitk.Resample(
        moving_image_resized,
        fixed_image_resized,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image_resized.GetPixelID()
    )

    # Apply colormap to the thermal image to show thermal variations
    moving_resampled_np = sitk.GetArrayFromImage(moving_resampled)

    moving_resampled_np = cv2.normalize(moving_resampled_np, None, 0, 255, cv2.NORM_MINMAX)
    moving_resampled_np = 255 - moving_resampled_np  # Invert the thermal image
    moving_resampled_colored = cv2.applyColorMap(moving_resampled_np.astype(np.uint8), cv2.COLORMAP_JET)
    print(moving_resampled_colored.shape)

    # Display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_image_cv_resized)
    plt.title('Fixed Image')
    plt.subplot(1, 2, 2)
    plt.imshow(moving_resampled_colored)
    plt.title('Registered Moving Image')
    plt.show()



    # Combine the thermal and optical images into a single 5-band image
    overlay = fixed_image_cv_resized.astype(np.float32)
    output = moving_resampled_colored.astype(np.float32)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    # Ensure the combined image is within the correct range
    output = np.clip(output, 0, 255).astype(np.uint8)

    #crop
    x_min, y_min, x_max, y_max = find_bounding_box(moving_resampled_colored)
    cropped_output = output[y_min:y_max, x_min:x_max]

    # Check the size of the combined image
    print("Combined image shape:", output.shape)
    print("Combined  cropped image shape:", cropped_output.shape)

    height, width, _ = output.shape

    # Upscale the combined image using super-resolution
    #sr = dnn_superres.DnnSuperResImpl_create()
    #path = r"C:\Users\TEK1\OneDrive - Syracuse University\syracuse university\Thesis\pictures\EDSR_x3.pb"  # Path to the pre-trained super-resolution model
    #sr.readModel(path)
    #sr.setModel("edsr", 3)  # You can use other models like "espcn", "fsrcnn", and "lapsrn"

    #upscaled_output = sr.upsample(output)

    # Ensure the upscaled image is within the correct range
    #upscaled_output = np.clip(upscaled_output, 0, 255).astype(np.uint8)

    # Save and display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Combined 5-Band Image')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_output, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.show()
    print(cropped_output.ndim)
    cv2.imwrite(
        r"C:\Users\TEK1\OneDrive - Syracuse University\syracuse university\Thesis\pictures\final 5 band image\final4.jpg",
        output)
    cv2.imwrite(
        r"C:\Users\TEK1\OneDrive - Syracuse University\syracuse university\Thesis\pictures\final 5 band image\cropped2.jpg",
        cropped_output)
    return final_transform

# crop
def find_bounding_box(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plt.title('Binary Mask')
    plt.imshow(thresh, cmap='gray')
    plt.show()

    print("Number of contours found = {}".format(len(contours)))

    if len(contours) == 0:
        # No contours found, return the full image
        return 0, 0, image.shape[1], image.shape[0]
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    return int(x_min), int(y_min), int(x_max), int(y_max)




# Paths to the fixed (thermal) and moving (optical) images
fixed_image_path = r'C:\Users\TEK1\OneDrive - Syracuse University\syracuse university\Thesis\pictures\June28\2024_06_28_04_47_50_PM.jpg'
moving_image_path = r'C:\Users\TEK1\OneDrive - Syracuse University\syracuse university\Thesis\pictures\June28\IMG_0014.pgm'

# Convert the PGM image to JPG with proper normalization
image = cv2.imread(moving_image_path, cv2.IMREAD_UNCHANGED)

# Normalize the image to ensure proper intensity values
image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
image_normalized = image_normalized.astype(np.uint8)  # Convert to uint8

jpg_path = os.path.splitext(moving_image_path)[0] + '.jpg'
cv2.imwrite(jpg_path, image_normalized)

# Perform the registration and get the final transform
transform = mutual_information_registration(fixed_image_path, jpg_path)


import cProfile
import pstats
from pstats import SortKey

# Profile the mutual_information_registration function
cProfile.run('mutual_information_registration(fixed_image_path, jpg_path)', 'restats')

# Create a Stats object
p = pstats.Stats('restats')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)  # Display top 10 results
