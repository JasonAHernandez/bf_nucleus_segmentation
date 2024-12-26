from PIL import Image, ImageDraw, UnidentifiedImageError
import imageio
import roifile
import os
import re


class MaskCreator:
    def __init__(self, image_dir, roi_dir, output_dir, image_indicator, roi_indicator, max_size=None, subtract_value=0):
        self.image_dir = image_dir
        self.roi_dir = roi_dir
        self.output_dir = output_dir
        self.image_indicator = image_indicator  # The part of the filename to replace
        self.roi_indicator = roi_indicator  # The replacement part
        self.max_size = max_size
        self.subtract_value = subtract_value  # Value to subtract before dividing by 2

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Define a regex pattern to match image names and indices
        self.pattern = re.compile(rf"^(.*){self.image_indicator}(\d{{3}})\.tif$")

    def create_mask_from_roi(self, image_path, roi_path, output_path):
        # Load the TIFF image to get dimensions
        image = imageio.imread(image_path)
        height, width = image.shape[:2]

        # Create a blank (black) mask with the same dimensions as the image
        mask = Image.new("L", (width, height), 0)

        # Load the ROI file and get the coordinates
        roi = roifile.ImagejRoi.fromfile(roi_path)
        coords = roi.coordinates()  # Coordinates of the ROI points

        # Ensure coordinates are integers
        coords = [(int(x), int(y)) for x, y in coords]

        # Draw a filled polygon based on the ROI coordinates
        draw = ImageDraw.Draw(mask)
        draw.polygon(coords, outline=255, fill=255)  # Outline and fill with white (255)

        # Save the mask with the exact same filename as the original image
        mask.save(output_path)

    def process_images(self):
        # Loop through the images
        for image_filename in os.listdir(self.image_dir):
            if image_filename.endswith(".tif"):
                image_path = os.path.join(self.image_dir, image_filename)

                # Attempt to open the image to check its dimensions, skipping if unidentifiable
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width > self.max_size or height > self.max_size:
                            continue
                except UnidentifiedImageError:
                    print(f"Skipping unidentifiable image file: {image_filename}")
                    continue

                # Match the filename with the pattern
                match = self.pattern.match(image_filename)
                if match:
                    base_name, index_str = match.groups()

                    # Convert index to integer, adjust based on the new logic, and format as a 3-digit string
                    index = int(index_str)
                    roi_index = f"{(index - self.subtract_value) // 2:03}"  # Subtract custom value and divide by 2
                    # roi_index = f"{(index - self.subtract_value) :03}" # without dividing by 2

                    # Construct paths
                    roi_filename = f"{base_name}{self.roi_indicator}{roi_index}.tif.roi"  # Include .tif before .roi
                    roi_path = os.path.join(self.roi_dir, roi_filename)
                    output_path = os.path.join(self.output_dir, image_filename)  # Use the exact image filename for the mask

                    # Print the current ROI file name for verification
                    print(f"Processing ROI file: {roi_filename}")

                    # Check if the ROI file exists
                    if os.path.exists(roi_path):
                        self.create_mask_from_roi(image_path, roi_path, output_path)
                        print(f"Saved mask for {image_filename} to {output_path}")
                    else:
                        print(f"ROI file {roi_filename} not found for image {image_filename}")
