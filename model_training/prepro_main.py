from apply_roi_mask import MaskCreator

if __name__ == '__main__':
    # Define directories
    image_dir = r"path\to\brightfield\images"  # Folder containing input images
    roi_dir = r"path\to\RoiSet"  # Folder containing ROI files
    output_dir = r"output\folder\for\where\masks\should\be\saved"  # Folder to save output masks

    # Define parameters
    image_indicator = "bf"  # The part to be replaced
    roi_indicator = "_RPE1_H1.2_"  # The replacement part
    max_size = 370
    subtract_value = 80

    # Create an instance of MaskCreator
    mask_creator = MaskCreator(
        image_dir=image_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        image_indicator=image_indicator,
        roi_indicator=roi_indicator,
        max_size=max_size,
        subtract_value=subtract_value
    )

    # Process the images to generate masks
    mask_creator.process_images()
