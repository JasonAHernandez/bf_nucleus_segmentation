from NucleusSegmentationModel import NucleusSegmentationModel
from tensorflow.keras.callbacks import ReduceLROnPlateau


# building a model with multiple folders
if __name__ == '__main__':
    model = NucleusSegmentationModel(input_size=(128, 128, 3), backbone='resnet34',
                                      initial_lr=1e-7, target_lr=1e-5, warmup_epochs=5)
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.9, patience=3, min_lr=1e-8, verbose=1
    )
    brightfield_folders = [
        r"C:\path\to\brightfield\images\bf_h1_2",
        r"C:\path\to\brightfield\images\bf_h2b"
    ]
    mask_folders = [
        r"C:\path\to\respective\bf\masks\masks_h1_2",
        r"C:\path\to\respective\bf\masks\masks_h2b"
    ]

    # Step 3: Load images and masks from all folders
    train_images, train_masks, val_images, val_masks = model.load_images(brightfield_folders, mask_folders)

    #in the save_dir directory given below these images will be saved with a prediction after each training epoch so you can see the training over time
    test_image1 = train_images[0]
    test_image2 = train_images[1]
    test_image3 = train_images[2]
    test_image4 = train_images[3]
    processed_image1 = model.preprocess_input(test_image1)
    processed_image2 = model.preprocess_input(test_image2)
    processed_image3 = model.preprocess_input(test_image3)
    processed_image4 = model.preprocess_image(test_image4)
    save_dir = r"C:\path\to\where\you\want\training\prediction\updates\epochs"
    
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
    
    # Step 6: Save the trained model (optional)
    model.save_model(r"C:\path\to\where\you\want\model\saved\to\RN34_NSM_hela_V1.keras")
                                                                # I have been using a name with the model type RN34 and the cell types used 'hela'

    # saving log of run
    # this creates a log which has the current NucleusSegmentationModel.py and current NSM_main.py which is mostly used so I can have a log of 
    # every parameter I used for the model while figuring out which parameters work best

    # log_file_path = r"C:\path\to\where\you\want\log\saved\to\RN34_NSM_hela_V1.txt"
    # purpose = "Creating a unet model for HeLa S3 cells to segment out the nucleus."
    # script_path = r"C:\path\to\this\file\NSM_main.py"
    # model.create_run_log(log_file_path, purpose, script_path)


