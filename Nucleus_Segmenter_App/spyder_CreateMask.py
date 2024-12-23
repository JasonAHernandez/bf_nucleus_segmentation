import os
import cv2
import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from segmentation_models import get_preprocessing
from tifffile import imread as tif_imread


class CreateMask:
    """
    A class to create and optionally verify masks for images using a trained model.
    """

    def __init__(self, model, mask_save_folder: str, backbone='resnet34', expected_size=(370, 370), input_size=(128, 128, 3)):
        """
        Initialize the CreateMask class.

        :param model: Trained model used to predict masks.
        :param mask_save_folder: Directory to save the generated and modified masks.
        :param expected_size: Expected size of the input images for mask creation (width, height).
        """
        self.model = model
        self.backbone = backbone
        self.preprocess_input = get_preprocessing(self.backbone)
        self.mask_save_folder = mask_save_folder
        self.expected_size = expected_size
        self.input_size = input_size

        os.makedirs(self.mask_save_folder, exist_ok=True)

        # Create log folder as a sibling to the mask_save_folder
        self.log_folder = os.path.join(os.path.dirname(self.mask_save_folder), "log_folder")
        os.makedirs(self.log_folder, exist_ok=True)
        self.log_file_path = os.path.join(self.log_folder, "log_of_skipped_bf_images.txt")

    def load_image(self, image_path):
        """
        Load an image from the specified path, handling TIFF and JPG files independently.

        :param image_path: Path to the image file.
        :return: Loaded and normalized image as a numpy array.
        """
        if image_path.lower().endswith(('.tif', '.tiff')):
            image = tif_imread(image_path)
            if image.dtype != np.uint8:
                image = (image / np.max(image) * 255).astype(np.uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3 and image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.ndim == 3 and image.shape[-1] > 3:
                image = image[:, :, :3]
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = load_img(image_path, target_size=self.input_size)
            image = img_to_array(image)
        else:
            raise ValueError(f"Unsupported file format for {image_path}")
        image = np.array(Image.fromarray(image.astype(np.uint8)).resize(self.input_size[:2]))
        return image / 255.0

    def preprocess_image(self, image):
        """
        Preprocess an image to align with NucleusSegmentationModel's expectations.

        :param image: Input image as a numpy array.
        :return: Preprocessed image array.
        """
        if image.dtype != np.uint8:
            image = (image / np.max(image) * 255).astype(np.uint8)  # Scale to uint8 range

        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:  # Single-channel image
            image = np.repeat(image, 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] > 3:  # Truncate excess channels
            image = image[:, :, :3]

        # Resize to match model input size
        resized_image = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        return resized_image / 255.0

    def postprocess_image(self, output, original_size):
        """
        Resize the model's output (mask) back to the original image size.

        :param output: Model's predicted mask.
        :param original_size: Original image dimensions (width, height).
        :return: Resized mask to match the original image size.
        """
        return cv2.resize(output, original_size, interpolation=cv2.INTER_NEAREST)

    def fit_oblong_shape(self, mask):
        """
        Adjust the predicted mask to fit an oblong-like shape.

        :param mask: Predicted mask as a NumPy array.
        :return: Adjusted mask as a binary NumPy array.
        """
        mask = (mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        adjusted_mask = np.zeros_like(mask)

        for contour in contours:
            if len(contour) > 5:  # Only process significant contours
                hull = cv2.convexHull(contour)
                cv2.drawContours(adjusted_mask, [hull], -1, 1, thickness=cv2.FILLED)

        return adjusted_mask

    def fill_in(self, mask):
        """
        Retain only the largest connected region (centermost ROI) in the mask, removing smaller disconnected regions.

        :param mask: Binary mask as a NumPy array.
        :return: Updated mask with only the central ROI retained.
        """
        mask = (mask > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype(np.uint8)

        return largest_component_mask

    def adjust_mask_size(self, mask, change=2):
        """
        Adjust the mask size by growing or shrinking the edges.

        :param mask: Input binary mask as a NumPy array.
        :param change: Positive value to grow the mask, negative to shrink it.
        :return: Modified binary mask.
        """
        kernel_size = abs(change)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if change > 0:
            return cv2.dilate(mask, kernel, iterations=1)
        else:
            return cv2.erode(mask, kernel, iterations=1)

    def draw_inner_edge(self, image, mask):
        """
        Overlay the inner edge of the mask on the original image.

        :param image: Original input image.
        :param mask: Binary mask as a NumPy array.
        :return: Image with edge overlay.
        """
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.uint8)

        # Match mask size to image size
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Properly scale uint16 images
        if image.dtype == np.uint16:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Convert grayscale image to RGB
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):
            edge_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            edge_image = image.copy()

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(edge_image, contours, -1, (0, 0, 0), 2)  # Black edge
        else:
            print("No contours found; skipping drawing step.")

        return edge_image

    def plot_verification(self, original_image, background_subtracted, edge_image):
        """
        Display the verification plot with three panels: original image, background subtracted, and edge.

        :param original_image: Original input image.
        :param background_subtracted: Image with mask applied.
        :param edge_image: Image with inner edge overlay.
        """
        # Adjust background_subtracted for consistent brightness
        if background_subtracted.dtype != np.uint8:
            background_subtracted = (background_subtracted / background_subtracted.max() * 255).astype(np.uint8)

        plt.figure(figsize=(12, 4))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image, cmap="gray")
        plt.axis("off")

        # Background Subtracted
        plt.subplot(1, 3, 2)
        plt.title("Background Subtracted")
        plt.imshow(background_subtracted, cmap="gray")
        plt.axis("off")

        # Edge Image
        plt.subplot(1, 3, 3)
        plt.title("Inner Edge")
        plt.imshow(edge_image, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def verify_mask(self, original_image, current_mask):
        """
        Verify the mask and allow for optional interactive adjustment of edge points.

        :param original_image: Original input image.
        :param current_mask: Generated mask as a NumPy array.
        :return: Verified mask or None if skipped.
        """
        print("Verification mode is active.")
        while True:
            # Fit the mask to an oblong shape and fill the interior
            oblong_mask = self.fit_oblong_shape(current_mask)
            oblong_mask = self.fill_in(oblong_mask)

            # Create the edge overlay image
            background_subtracted = cv2.bitwise_and(original_image, original_image, mask=oblong_mask)
            edge_image = self.draw_inner_edge(original_image, oblong_mask)

            # Display the images for verification
            self.plot_verification(original_image, background_subtracted, edge_image)

            # Present options to the user after showing images
            print("Options:")
            print(" - Press 'i' to interactively adjust the mask edge.")
            print(" - Press 's' to save the mask as is.")
            print(" - Press 'g' to grow the mask.")
            print(" - Press 'r' to reduce the mask.")
            print(" - Press 'k' to skip this image.")
            choice = input("Your choice: ").strip().lower()

            if choice == 'i':
                # Generate edge points for interactive adjustment
                contours, _ = cv2.findContours(oblong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Select evenly spaced points along the largest contour
                    edge_points = contours[0][::max(1, len(contours[0]) // 30)].squeeze()
                    if len(edge_points.shape) != 2:  # Handle edge cases where contours are too small
                        edge_points = np.expand_dims(edge_points, axis=0)

                    # Launch interactive editing
                    print("Launching interactive edge adjustment...")
                    adjusted_points = interactive_edit(edge_image, edge_points)

                    # Recreate the mask based on adjusted points
                    oblong_mask = np.zeros_like(oblong_mask)
                    cv2.fillPoly(oblong_mask, [adjusted_points.astype(int)], 1)

                    # Update the current mask for continuity
                    current_mask = oblong_mask
                    print("Mask updated with adjusted points.")
                else:
                    print("No contours found for interactive editing.")

            elif choice == 's':
                print("Mask saved.")
                return oblong_mask
            elif choice == 'g':
                current_mask = self.adjust_mask_size(current_mask, change=5)
                print("Mask grown.")
            elif choice == 'r':
                current_mask = self.adjust_mask_size(current_mask, change=-5)
                print("Mask reduced.")
            elif choice == 'k':
                print("Image skipped.")
                return None
            else:
                print("Invalid input. Please enter 'i', 's', 'g', 'r', or 'k'.")

    def create_mask(self, tif_folder_path, verify=False):
        """
        Generate masks for all .tif images in a folder.

        :param tif_folder_path: Path to the folder containing .tif images.
        :param verify: Whether to verify masks interactively.
        """
        skipped_images = []
        for filename in os.listdir(tif_folder_path):
            image_path = os.path.join(tif_folder_path, filename)

            if not filename.endswith(".tif"):
                print(f"Skipped non-TIF file: {filename}")
                continue

            try:
                original_image = tifffile.imread(image_path)
            except Exception as e:
                print(f"Error reading {filename}, skipping: {e}")
                continue

            # Check if the image dimensions match the expected size
            if original_image.shape[:2] != self.expected_size:
                print(
                    f"Skipped {filename}: Image size {original_image.shape[:2]} does not match expected size {self.expected_size}.")
                continue

            print(f"Processing image: {filename}")
            preprocessed_image = self.preprocess_image(original_image)

            predicted_mask = self.model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
            resized_mask = self.postprocess_image(predicted_mask, original_image.shape[:2])
            oblong_mask = self.fit_oblong_shape(resized_mask)
            oblong_mask = self.fill_in(oblong_mask)

            if verify:
                verified_mask = self.verify_mask(original_image, oblong_mask)
                if verified_mask is None:
                    skipped_images.append(filename)
                    continue
            else:
                verified_mask = oblong_mask

            mask_save_path = os.path.join(self.mask_save_folder, filename.replace('.tif', '.tif'))
            Image.fromarray((verified_mask * 255).astype(np.uint8)).save(mask_save_path)
            print(f"Mask saved: {mask_save_path}")

        if skipped_images:
            with open(self.log_file_path, 'w') as log_file:
                log_file.write("\n".join(skipped_images))
            print(f"Skipped images logged to {self.log_file_path}")
