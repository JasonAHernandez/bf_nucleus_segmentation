from apply_roi_mask import MaskCreator

if __name__ == '__main__':
    # Define directories
    image_dir = r"path\to\brightfield\images"  # Folder containing input images
    roi_dir = r"path\to\RoiSet"  # Folder containing ROI files
    output_dir = r"output\folder\for\where\masks\should\be\saved"  # Folder to save output masks

    # Define parameters
    roi_indicator = "roi"  # Indicator to replace 'bf' in ROI filenames
    max_size = 370  # Maximum size of images to process
    subtract_value = 80  # Subtraction value for ROI index calculation

    # Create an instance of MaskCreator
    mask_creator = MaskCreator(
        image_dir=image_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        roi_indicator=roi_indicator,
        max_size=max_size,
        subtract_value=subtract_value
    )

    # Process the images to generate masks
    mask_creator.process_images()
