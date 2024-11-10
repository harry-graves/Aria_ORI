import projectaria_tools.core.mps as mps
import numpy as np
import csv
import cv2
import os
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze
)

def draw_cross_on_image(image, pixel_x, pixel_y, cross_color=(0, 0, 255), cross_size=10, cross_thickness=2):
    """
    Draws a cross on the given image at the specified pixel coordinates.
    """
    # Ensure image is a contiguous array for OpenCV compatibility
    image = np.ascontiguousarray(image)

    # Check coordinates
    print(f"Drawing cross at: ({pixel_x}, {pixel_y})")

    # Draw horizontal and vertical lines to make a cross
    cv2.line(image, (int(pixel_x) - cross_size, int(pixel_y)), 
             (int(pixel_x) + cross_size, int(pixel_y)), cross_color, cross_thickness)
    cv2.line(image, (int(pixel_x), int(pixel_y) - cross_size), 
             (int(pixel_x), int(pixel_y) + cross_size), cross_color, cross_thickness)
    
    return image
    
def images_from_indexes(vrs_file, gaze_path, output_dir):

    # Initialize data provider
    print(f"Creating data provider from {vrs_file}")
    provider = data_provider.create_vrs_data_provider(vrs_file)
    if not provider:
        print("Invalid vrs data provider")

    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    pinhole = calibration.get_linear_camera_calibration(512, 512, 150, camera_label,calib.get_transform_device_camera())
    device_calibration = provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(camera_label)

    gaze_cpf = mps.read_eyegaze(gaze_path)

    total_images = provider.get_num_data(stream_id)
    start_index = 100

    for i in range(start_index, total_images):

        # Get raw image data
        image_data = provider.get_image_data_by_index(stream_id, i)
        image_array = image_data[0].to_numpy_array()
        timestamp = image_data[1].capture_timestamp_ns

        # Apply undistortion to the image
        rectified_array = calibration.distort_by_calibration(image_array, pinhole, calib)
        rotated_array = np.rot90(rectified_array, k=3)

        # Get raw eye-gaze data
        eye_gaze_info = get_nearest_eye_gaze(gaze_cpf, timestamp)
        gaze_projection = get_gaze_vector_reprojection(eye_gaze_info, camera_label, device_calibration, rgb_camera_calibration, depth_m = 1.0)
        pixel_x = int(gaze_projection[0])
        pixel_y = int(gaze_projection[1])

        # Apply undistortion to the eye-gaze point
        distorted_gaze = np.zeros((1408,1408,3))
        
        # Set area around the ey-gaze point to be an arbitrary value < 255
        # Must be a 5x5 area because distort_by_calibration compresses from 1408x1408 to 512x512
        distorted_gaze[pixel_x - 2 : pixel_x + 2,pixel_y - 2 : pixel_y + 2 , 0] = 255
        rectified_gaze = calibration.distort_by_calibration(distorted_gaze, pinhole, calib)
        rotated_gaze = np.rot90(rectified_gaze, k=1)
        rotated_gaze = np.sum(rotated_gaze, axis=2)

        if np.count_nonzero(rotated_gaze) == 0:
            print("No non-zero point found in rectified and rotated gaze.")
            continue

        final_gaze = np.argwhere(rotated_gaze != 0)
        pixel_x = final_gaze[0][0]
        pixel_y = final_gaze[0][1]

        image = draw_cross_on_image(rotated_array,pixel_x,pixel_y)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'image_{i}.png')
        cv2.imwrite(save_path,image)


# Initialize directories
vrs_file = "data/ET1.vrs"
gaze_path = "data/eye_gaze/general_eye_gaze.csv"
output_dir = "data/gaze_images"

images_from_indexes(vrs_file, gaze_path, output_dir)