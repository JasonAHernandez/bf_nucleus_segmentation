from apply_roi_mask import MaskCreator

if __name__ == '__main__':
    # Define directories
    image_dir = r"path\to\brightfield\images"  # Folder containing input images
    roi_dir = r"path\to\RoiSet"  # Folder containing ROI files
    output_dir = r"output\folder\for\where\masks\should\be\saved"  # Folder to save output masks

    # Define parameters
    image_indicator = "bf"  # The part to be replaced
    roi_indicator = "_RPE1_H1.2_"  # The replacement part

    # Define a custom ROI index formula
    roi_index_formula = lambda index: (index - 80) // 2  # Example: Subtract 80 and multiply by 3

    # Create an instance of MaskCreator
    mask_creator = MaskCreator(
        image_dir=image_dir,
        roi_dir=roi_dir,
        output_dir=output_dir,
        image_indicator=image_indicator,
        roi_indicator=roi_indicator,
        roi_index_formula=roi_index_formula
    )

    # Process the images to generate masks
    mask_creator.process_images()
