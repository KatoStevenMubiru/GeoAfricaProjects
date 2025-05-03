# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# --- Configuration ---
# Default Input Directory (where 'all_cultural_images_collected' style folders are)
DEFAULT_INPUT_DIR = "BingImages"
# Default Output Directory (where blurred images will be saved)
DEFAULT_OUTPUT_DIR = "BingImagesBlurred"
# Face Blurring Settings
PROTOTXT_PATH = "deploy.prototxt.txt" # Path to the Caffe prototxt file
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel" # Path to the Caffe model file
CONFIDENCE_THRESHOLD = 0.3 # Minimum probability to consider a detection a face
BLUR_KERNEL_SIZE = (45, 45) # Size of the Gaussian blur kernel (odd numbers)

# --- Global variable for the loaded network ---
face_net = None

def initialize_face_detector():
    """Loads the face detection model into memory."""
    global face_net
    if face_net is None:
        if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
            print(f"Error: Face detection model files not found.")
            print(f"  Expected prototxt: {os.path.abspath(PROTOTXT_PATH)}")
            print(f"  Expected caffemodel: {os.path.abspath(MODEL_PATH)}")
            print("  Cannot proceed without the model files.")
            return False
        try:
            print("Loading face detection model...")
            face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
            print("Face detection model loaded successfully.")
            return True
        except cv2.error as e:
            print(f"Error loading Caffe model: {e}")
            return False
        except Exception as e:
             print(f"An unexpected error occurred loading the face model: {e}")
             return False
    return True # Already loaded

def blur_image_faces(input_image_path, output_image_path):
    """
    Reads an image, detects faces, blurs them, and saves to the output path.
    Returns True if successful or no faces found, False otherwise.
    """
    global face_net
    if face_net is None:
        print("Error: Face detector not initialized.")
        return False

    try:
        # Read the image using OpenCV
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Warning: Could not read image {input_image_path}. Skipping.")
            return True # Skip this file, but don't count as failure

        (h, w) = image.shape[:2]

        # Create a blob from the image and pass it through the network
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        faces_blurred = 0
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > CONFIDENCE_THRESHOLD:
                faces_blurred += 1
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box stays within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Extract the face ROI
                face = image[startY:endY, startX:endX]

                # Apply Gaussian blur to the face ROI
                # Check if face ROI is valid before blurring
                if face.size > 0:
                    blurred_face = cv2.GaussianBlur(face, BLUR_KERNEL_SIZE, 0)
                    # Put the blurred face back into the original image
                    image[startY:endY, startX:endX] = blurred_face
                else:
                     print(f"Warning: Invalid face ROI detected in {input_image_path} at index {i}. Skipping blur for this detection.")


        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_image_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save the resulting image (with blurred faces)
        cv2.imwrite(output_image_path, image)
        # if faces_blurred > 0:
        #     print(f"  Blurred {faces_blurred} face(s) in {os.path.basename(input_image_path)}")
        return True

    except cv2.error as cv_e:
         print(f"OpenCV Error processing {input_image_path}: {cv_e}")
         return False
    except Exception as e:
        print(f"Error processing image {input_image_path}: {e}")
        traceback.print_exc()
        return False

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and blur faces in images.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_DIR,
                        help="Path to the input directory containing culture folders.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR,
                        help="Path to the output directory where blurred images will be saved.")
    args = parser.parse_args()

    input_dir = args.input
    output_dir_base = args.output

    print("--- Starting Face Blurring Process ---")
    print(f"Input Directory:  {os.path.abspath(input_dir)}")
    print(f"Output Directory: {os.path.abspath(output_dir_base)}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        exit()

    # Initialize the face detector
    if not initialize_face_detector():
        exit()

    # --- Collect all image paths ---
    all_image_paths = []
    print("Scanning for images...")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                all_image_paths.append(os.path.join(root, filename))

    print(f"Found {len(all_image_paths)} images to process.")

    # --- Process images ---
    processed_count = 0
    failed_count = 0
    skipped_count = 0

    # Use tqdm for a progress bar
    for input_path in tqdm(all_image_paths, desc="Blurring Faces"):
        try:
            # Determine the corresponding output path
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir_base, relative_path)

            success = blur_image_faces(input_path, output_path)
            if success:
                processed_count += 1
            else:
                # If blur_image_faces returned False explicitly due to error
                failed_count += 1
                print(f"Failed to process: {input_path}") # Log failure

        except Exception as e:
            failed_count += 1
            print(f"Critical error processing {input_path}: {e}")
            traceback.print_exc()


    print("\n--- Blurring Summary ---")
    print(f"Successfully processed (or skipped unreadable): {processed_count}")
    print(f"Failed during processing: {failed_count}")
    print(f"Total images scanned: {len(all_image_paths)}")
    print(f"Blurred images saved to: {os.path.abspath(output_dir_base)}")
    print("\n===== Script Finished =====")
