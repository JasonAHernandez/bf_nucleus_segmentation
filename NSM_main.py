from NucleusSegmentationModel import NucleusSegmentationModel
from tensorflow.keras.callbacks import ReduceLROnPlateau


# building a model with multiple folders
if __name__ == '__main__':
    model = NucleusSegmentationModel(input_size=(128, 128, 3), backbone='resnet34',
                                      initial_lr=1e-7, target_lr=1e-5, warmup_epochs=5)
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.9, patience=3, min_lr=1e-8, verbose=1
    )
    # brightfield_folders = [
    #     r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\bf_h1_2",
    #     r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\bf_h2b",
    #     r"C:\Users\jason\Documents\jupyter_projects\nucleus_finding\tif_cell_images"
    # ]
    #
    # mask_folders = [
    #     r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\masks_h1_2",
    #     r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\masks_h2b",
    #     r"C:\Users\jason\Documents\jupyter_projects\nucleus_finding\tif_masks"
    # ]

    # brightfield_folders = [
    #     r"C:\Users\jason\Documents\jupyter_projects\nucleus_finding\tif_cell_images"
    # ]
    #
    # mask_folders = [
    #     r"C:\Users\jason\Documents\jupyter_projects\nucleus_finding\tif_masks"
    # ]

    brightfield_folders = [
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\cs_projects\nucleus_segmentation\tif_cell_images",
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\bf_h1_2",
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\bf_h2b"
    ]

    mask_folders = [
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\cs_projects\nucleus_segmentation\tif_masks",
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\masks_h1_2",
        r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\students\masa\bf_images\BF_roi\masks_h2b"
    ]

    # Step 3: Load images and masks from all folders
    train_images, train_masks, val_images, val_masks = model.load_images(brightfield_folders, mask_folders)
    # train_images, train_masks = model.load_images(brightfield_folders, mask_folders) # for current best model

    # Step 4: Split data into training and validation sets
    # train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.5, random_state=42)

    test_image1 = train_images[0]
    test_image2 = train_images[50]
    test_image3 = train_images[100]
    test_image4 = train_images[150]
    processed_image1 = model.preprocess_input(test_image1)
    processed_image2 = model.preprocess_input(test_image2)
    processed_image3 = model.preprocess_input(test_image3)
    processed_image4 = model.preprocess_image(test_image4)
    save_dir = r"C:\Users\jason\PycharmProjects\nucleus_outline\unet\models\rn34\final\epochs"
    # Step 5: Train the model
    history = model.train_model(
        train_images, train_masks,
        val_images, val_masks,
        epochs=70, batch_size=4,
        sample_images=[(test_image1, processed_image1),
                       (test_image2, processed_image2),
                       (test_image3, processed_image3),
                       (test_image4, processed_image4)],
        save_dir=save_dir, iterations=1, callbacks=[lr_callback])

    # history = model.train_model(train_images, train_masks, val_images, val_masks, epochs=50, batch_size=8) # for current best model

    # Step 6: Save the trained model (optional)
    model.save_model(r"C:\Users\jason\PycharmProjects\nucleus_outline\unet\models\rn34\final\RN34_NSM_hela_rpe1_V1.keras")

    #saving log of run
    log_file_path = r"C:\Users\jason\PycharmProjects\nucleus_outline\unet\models\rn34\final\RN34_NSM_hela_rpe1_V1.txt"
    purpose = "Creating a unet model for HeLa S3 cells to segment out the nucleus."
    script_path = r"/unet/uint16/NSM_main.py"
    model.create_run_log(log_file_path, purpose, script_path)


