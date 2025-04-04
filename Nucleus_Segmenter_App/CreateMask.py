import os
import cv2
import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from segmentation_models import get_preprocessing
from tifffile import imread as tif_imread
import nd2reader
import re


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

    def load_first_frame_from_tif(self, movie_path):
        """
        Loads the first frame from a multi-frame .tif movie file.

        :param movie_path: Path to the .tif movie file.
        :return: First frame as a NumPy array.
        """
        try:
            with tifffile.TiffFile(movie_path) as tif:
                first_frame = tif.pages[0].asarray()  # Extract the first frame
                first_frame = (first_frame / np.max(first_frame) * 255).astype(np.uint8)  # Normalize to uint8
            return first_frame
        except Exception as e:
            print(f"Error reading {movie_path}: {e}")
            return None

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

    def draw_inner_edge(self, image, mask, color=(0, 0, 0)):
        """
        Overlay the inner edge of the mask on the original image.

        :param image: Original input image.
        :param mask: Binary mask as a NumPy array.
        :param color: Edge color in (B, G, R) format.
        :return: Image with edge overlay.
        """
        mask = (mask > 0.5).astype(np.uint8)

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if image.dtype == np.uint16:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):
            edge_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            edge_image = image.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(edge_image, contours, -1, color, 2)  # Custom color for edges
        else:
            print("No contours found; skipping drawing step.")

        return edge_image

    def plot_verification(self, bf_image, bf_bg_subtracted, bf_edge_image,
                          movie_image=None, movie_bg_subtracted=None, movie_edge_image=None):
        """
        Display the verification plot with panels for BF and movie images.

        :param bf_image: Brightfield input image.
        :param bf_bg_subtracted: BF image with mask applied.
        :param bf_edge_image: BF image with inner edge overlay.
        :param movie_image: First frame from cleaned movie.
        :param movie_bg_subtracted: Movie frame with mask applied.
        :param movie_edge_image: Movie frame with edge overlay.
        """
        plt.figure(figsize=(12, 6))

        # Brightfield images
        plt.subplot(2, 3, 1)
        plt.title("BF Image")
        plt.imshow(bf_image, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("BF Mask Applied")
        plt.imshow(bf_bg_subtracted, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("BF Mask Outline")
        plt.imshow(bf_edge_image, cmap="gray")
        plt.axis("off")

        # Movie images if available
        if movie_image is not None:
            plt.subplot(2, 3, 4)
            plt.title("Movie Frame")
            plt.imshow(movie_image, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.title("Movie Mask Applied")
            plt.imshow(movie_bg_subtracted, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.title("Movie Mask Outline (Red)")
            plt.imshow(movie_edge_image, cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def extract_index(self, filename):
        """
        Extracts the numeric index from a filename based on the last underscore `_`
        before the file extension.

        Example:
            "2025-02-15_HeLaS3_H3-2-Halo_c25_RH_001.tif"  -> "001"
            "2025-02-15_HeLaS3_H3-2-Halo_c25_RH_002.nd2"  -> "002"

        :param filename: Name of the file.
        :return: Extracted index as a string, or None if no match.
        """
        match = re.search(r'_([^_]+)\.[^.]+$', filename)  # Extract text after last "_" before the file extension
        if match:
            return match.group(1)  # Return the matched index
        else:
            print(f"Warning: Could not extract index from {filename}")
            return None

    def verify_mask(self, original_image, current_mask, movie_frame=None):
        """
        Verify the mask and allow interactive adjustment using both the brightfield image and the first frame of a cleaned .tif movie.

        :param original_image: Brightfield input image.
        :param current_mask: Generated mask as a NumPy array.
        :param movie_frame: First frame from cleaned .tif movie (optional).
        :return: Verified mask or None if skipped.
        """
        print("Verification mode is active.")

        # Contrast enhancement settings (Modify these as needed)
        contrast_method = "clahe"
        clip_limit = 3.0
        gamma = 1.5

        user_adjusted = False  # Track if the user manually modified the mask

        while True:
            # Step 1: Process the mask only if user hasn't edited it
            if not user_adjusted:
                oblong_mask = self.fit_oblong_shape(current_mask)
                oblong_mask = self.fill_in(oblong_mask)
            else:
                oblong_mask = current_mask.copy()

            # Step 2: Background-subtracted images
            bf_background_subtracted = cv2.bitwise_and(original_image, original_image, mask=oblong_mask)
            movie_background_subtracted = None
            enhanced_movie_frame = None

            if movie_frame is not None:
                enhanced_movie_frame = enhance_image_contrast(
                    movie_frame, method=contrast_method, clip_limit=clip_limit, gamma=gamma
                )
                movie_background_subtracted = cv2.bitwise_and(enhanced_movie_frame, enhanced_movie_frame,
                                                              mask=oblong_mask)

            # Step 3: Overlay mask edge for visualization
            bf_edge_image = self.draw_inner_edge(original_image, oblong_mask, color=(0, 0, 0))
            movie_edge_image = None
            if movie_frame is not None:
                movie_edge_image = self.draw_inner_edge(enhanced_movie_frame, oblong_mask, color=(255, 0, 0))

            # Step 4: Show both BF and movie views
            self.plot_verification(
                original_image, bf_background_subtracted, bf_edge_image,
                enhanced_movie_frame, movie_background_subtracted, movie_edge_image
            )

            # Step 5: User input
            print("Options:")
            print(" - Press 's' to save the mask as is.")
            print(" - Press 'g' to grow the mask.")
            print(" - Press 'r' to reduce the mask.")
            print(" - Press 'k' to skip this image.")
            print(" - Press 'ib' to interactively edit using the BF image.")
            print(" - Press 'im' to interactively edit using the movie frame.")
            choice = input("Your choice: ").strip().lower()

            if choice == 'ib':
                contours, _ = cv2.findContours(oblong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    edge_points = self.evenly_spaced_contour_points(contours[0], num_points=30)
                    print("Launching interactive edge adjustment using BF image...")
                    adjusted_points = interactive_edit_dual_view(
                        edit_image=bf_edge_image,
                        reference_image=movie_edge_image if movie_edge_image is not None else bf_edge_image,
                        points=edge_points,
                        edit_title="Edit on Brightfield",
                        ref_title="Reference: Movie Frame",
                        edit_color='blue',
                        ref_color='red'
                    )
                    current_mask = np.zeros_like(oblong_mask)
                    cv2.fillPoly(current_mask, [adjusted_points.astype(int)], 1)
                    user_adjusted = True
                    print("Mask updated with adjusted points from BF image.")

            elif choice == 'im' and movie_frame is not None:
                contours, _ = cv2.findContours(oblong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    edge_points = self.evenly_spaced_contour_points(contours[0], num_points=30)
                    print("Launching interactive edge adjustment using movie frame...")
                    adjusted_points = interactive_edit_dual_view(
                        edit_image=movie_edge_image,
                        reference_image=bf_edge_image,
                        points=edge_points,
                        edit_title="Edit on Movie Frame",
                        ref_title="Reference: Brightfield",
                        edit_color='blue',
                        ref_color='black'
                    )
                    current_mask = np.zeros_like(oblong_mask)
                    cv2.fillPoly(current_mask, [adjusted_points.astype(int)], 1)
                    user_adjusted = True
                    print("Mask updated with adjusted points from movie frame.")

            elif choice == 's':
                print("Mask saved.")
                return oblong_mask

            elif choice == 'g':
                current_mask = self.adjust_mask_size(current_mask, change=5)
                user_adjusted = False
                print("Mask grown.")

            elif choice == 'r':
                current_mask = self.adjust_mask_size(current_mask, change=-5)
                user_adjusted = False
                print("Mask reduced.")

            elif choice == 'k':
                print("Image skipped.")
                return None

            else:
                print("Invalid input. Please enter 's', 'g', 'r', 'k', 'ib', or 'im'.")

    def create_mask(self, tif_folder_path, movie_folder_path=None, movie_index_formula="index", verify=False):
        """
        Generate masks for all .tif images in a folder and optionally overlay the mask on
        the first frame of a cleaned .tif movie.

        :param tif_folder_path: Path to the folder containing .tif images.
        :param movie_folder_path: Path to the folder containing cleaned .tif movie files (optional).
        :param movie_index_formula: User-defined transformation for converting BF index to movie index.
        :param verify: Whether to verify masks interactively.
        """
        skipped_images = []

        # Get all cleaned .tif movies and their extracted indices
        movie_index_map = {}
        if movie_folder_path:
            for movie_file in os.listdir(movie_folder_path):
                if movie_file.endswith(".tif") and "_cleaned" in movie_file:
                    movie_index = self.extract_index(movie_file.replace("_cleaned", ""))  # Remove "_cleaned" for matching
                    if movie_index is not None:
                        # print(movie_index)
                        movie_index_map[movie_index] = os.path.join(movie_folder_path, movie_file)

        for filename in os.listdir(tif_folder_path):
            if not filename.endswith(".tif"):
                print(f"Skipped non-TIF file: {filename}")
                continue

            image_path = os.path.join(tif_folder_path, filename)
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

            # Extract the index for this BF image
            bf_index = self.extract_index(filename)
            if bf_index is None:
                print(f"Skipped {filename}: Could not extract index.")
                continue

            # Apply the user-defined formula to transform the BF index into the movie index
            try:
                movie_index = str(eval(movie_index_formula, {"index": int(bf_index)}))  # Safely evaluate formula
                movie_index = movie_index.zfill(3)
            except Exception as e:
                print(f"Error applying movie index formula '{movie_index_formula}' on {bf_index}: {e}")
                continue

            movie_frame = None

            # Find the closest matching cleaned movie using the transformed movie index
            closest_movie = movie_index_map.get(movie_index)
            if closest_movie:
                movie_frame = self.load_first_frame_from_tif(closest_movie)
                print(f"Matched {filename} → {closest_movie} using formula: {movie_index_formula}")
            else:
                print(filename)
                print(closest_movie)
                print(movie_index_map.get(movie_index))
                print(movie_index_map)
                print(movie_index)
                print("found no matching movie for the bf image")
                exit()
            # Preprocess the image for the model
            preprocessed_image = self.preprocess_image(original_image)

            predicted_mask = self.model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
            resized_mask = self.postprocess_image(predicted_mask, original_image.shape[:2])
            oblong_mask = self.fit_oblong_shape(resized_mask)
            oblong_mask = self.fill_in(oblong_mask)

            if verify:
                verified_mask = self.verify_mask(original_image, oblong_mask, movie_frame)
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

    def evenly_spaced_contour_points(self, contour, num_points=30):
        """
        Interpolate `num_points` evenly spaced points along the given contour.

        :param contour: Input contour (Nx1x2 or Nx2 array of points).
        :param num_points: Number of points to sample.
        :return: (num_points, 2) array of evenly spaced points.
        """
        contour = contour.squeeze()

        # Compute cumulative arc length
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0)

        # Normalize cumulative distance to 0–1
        total_length = cumulative[-1]
        if total_length == 0:
            return np.repeat(contour[:1], num_points, axis=0)

        normalized = cumulative / total_length

        # Interpolate evenly spaced values between 0 and 1
        target_distances = np.linspace(0, 1, num_points)

        # Interpolate x and y separately
        x_interp = np.interp(target_distances, normalized, contour[:, 0])
        y_interp = np.interp(target_distances, normalized, contour[:, 1])

        return np.stack((x_interp, y_interp), axis=1)


class DraggablePoint:
    def __init__(self, point, plot, all_points, edge_plot, min_distance=2):
        self.point = point
        self.plot = plot
        self.all_points = all_points  # Reference to all points for updating the edge
        self.edge_plot = edge_plot  # Plot of the modified edge
        self.min_distance = min_distance  # Minimum allowed distance between points
        self.is_dragging = False

    def on_press(self, event):
        if event.inaxes == self.plot.axes:
            contains, _ = self.plot.contains(event)
            if contains:
                self.is_dragging = True

    def on_release(self, event):
        self.is_dragging = False

    def on_motion(self, event):
        if self.is_dragging:
            new_x, new_y = event.xdata, event.ydata
            for other_point in self.all_points:
                if other_point is not self.point:
                    # Ensure points don't overlap by enforcing a minimum distance
                    distance = np.sqrt((new_x - other_point[0])**2 + (new_y - other_point[1])**2)
                    if distance < self.min_distance:
                        print(f"Point too close to another point ({distance:.2f} < {self.min_distance}), ignoring drag.")
                        return

            # Update the point position
            self.point[0], self.point[1] = new_x, new_y
            self.plot.set_data(self.point[0], self.point[1])

            # Update the edge dynamically
            self.edge_plot.set_data(*zip(*self.all_points))
            self.plot.figure.canvas.draw()


def enhance_image_contrast(image, method="histogram", clip_limit=2.0, gamma=1.0):
    """
    Enhance the contrast of a grayscale image using different methods.

    :param image: Input grayscale image.
    :param method: Contrast enhancement method ('histogram', 'clahe', 'gamma').
    :param clip_limit: CLAHE contrast limit (higher = stronger enhancement).
    :param gamma: Gamma correction factor (lower <1 = darker, higher >1 = brighter).
    :return: Contrast-enhanced image.
    """
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Convert RGB to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == "histogram":
        enhanced_image = cv2.equalizeHist(image)  # Standard histogram equalization

    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)  # Adaptive histogram equalization (CLAHE)

    elif method == "gamma":
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        enhanced_image = cv2.LUT(image, table)  # Apply gamma correction

    else:
        print(f"Unknown contrast method '{method}', using default histogram equalization.")
        enhanced_image = cv2.equalizeHist(image)

    return enhanced_image

def interactive_edit(edge_image, points, enhance_contrast=False):
    """
    Launch an interactive GUI to adjust ROI edge points.

    :param edge_image: The image over which the edges are drawn.
    :param points: Initial list of points for the edge.
    :param enhance_contrast: Whether to apply contrast enhancement (for movie frames).
    :return: Updated list of points after interaction.
    """
    if enhance_contrast:
        edge_image = enhance_image_contrast(edge_image)  # Apply contrast enhancement

    fig, ax = plt.subplots()
    ax.imshow(edge_image, cmap="gray")
    ax.set_title("Adjust ROI Edge Points")

    # Original edge (static, black)
    original_edge_plot, = ax.plot(*zip(*points), 'k-', lw=2, alpha=0.7, label="Original Edge")

    # User-modified edge (dynamic, blue)
    modified_edge_plot, = ax.plot(*zip(*points), 'b-', lw=2, alpha=0.7, label="Modified Edge")

    # Draggable points
    plots = []
    draggable_points = []

    for point in points:
        plot, = ax.plot(point[0], point[1], 'ro')  # Red points for interaction
        draggable = DraggablePoint(point, plot, points, modified_edge_plot)
        plots.append(plot)
        draggable_points.append(draggable)

        fig.canvas.mpl_connect('button_press_event', draggable.on_press)
        fig.canvas.mpl_connect('button_release_event', draggable.on_release)
        fig.canvas.mpl_connect('motion_notify_event', draggable.on_motion)

    # Done button
    def on_done(event):
        plt.close(fig)

    done_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    done_button = Button(done_ax, 'Done')
    done_button.on_clicked(on_done)

    ax.legend(loc='upper right')
    plt.show()

    return np.array(points)

def interactive_edit_dual_view(edit_image, reference_image, points,
                               edit_title="Edit ROI", ref_title="Reference View",
                               edit_color='blue', ref_color='red'):
    """
    Interactive GUI to adjust ROI edge points with a side-by-side reference image.

    :param edit_image: The image to edit on.
    :param reference_image: The reference image (not editable).
    :param points: Initial ROI edge points.
    :param edit_title: Title for the editable image panel.
    :param ref_title: Title for the reference image panel.
    :param edit_color: Color for the editable edge line.
    :param ref_color: Color for the reference edge line.
    :return: Adjusted list of points.
    """
    fig, (ax_edit, ax_ref) = plt.subplots(1, 2, figsize=(12, 6))

    # Display editable image
    ax_edit.imshow(edit_image, cmap="gray")
    ax_edit.set_title(edit_title)
    editable_line, = ax_edit.plot(*zip(*points), color=edit_color, lw=2, label="Editable Edge")

    # Plot draggable points
    point_artists = []
    draggable_points = []
    for point in points:
        artist, = ax_edit.plot(point[0], point[1], 'ro')
        dp = DraggablePoint(point, artist, points, editable_line)
        point_artists.append(artist)
        draggable_points.append(dp)

        fig.canvas.mpl_connect('button_press_event', dp.on_press)
        fig.canvas.mpl_connect('button_release_event', dp.on_release)
        fig.canvas.mpl_connect('motion_notify_event', dp.on_motion)

    # Reference image with static edge
    ax_ref.imshow(reference_image, cmap="gray")
    ax_ref.set_title(ref_title)
    ax_ref.plot(*zip(*points), color=ref_color, lw=2, linestyle='--', label="Reference Edge")

    for ax in (ax_edit, ax_ref):
        ax.axis("off")
        ax.legend()

    # Done button
    def on_done(event):
        plt.close(fig)

    done_ax = plt.axes([0.45, 0.02, 0.1, 0.05])
    done_button = Button(done_ax, "Done")
    done_button.on_clicked(on_done)

    plt.tight_layout()
    plt.show()

    return np.array(points)




