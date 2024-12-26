import inspect
import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
from albumentations.augmentations.geometric.transforms import ElasticTransform
from albumentations.core.transforms_interface import ImageOnlyTransform
from segmentation_models import Unet, get_preprocessing
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tifffile import imread as tif_imread


@tf.keras.utils.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Tversky loss function to handle imbalance between false positives and false negatives.
    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return 1 - (true_pos + 1e-6) / (true_pos + alpha * false_neg + beta * false_pos + 1e-6)

@tf.keras.utils.register_keras_serializable()
def boundary_loss(y_true, y_pred):
    """
    Boundary-aware loss function that focuses on errors near object boundaries.
    Computes boundary maps using morphological operations in TensorFlow.

    :param y_true: Ground truth tensor.
    :param y_pred: Predicted tensor.
    :return: Boundary-aware loss value.
    """
    # Define a Sobel kernel for edge detection
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

    # Compute boundaries for y_true
    y_true = tf.cast(y_true, tf.float32)  # Ensure float type for convolution
    y_true_edge_x = tf.nn.conv2d(y_true, sobel_x, strides=[1, 1, 1, 1], padding="SAME")
    y_true_edge_y = tf.nn.conv2d(y_true, sobel_y, strides=[1, 1, 1, 1], padding="SAME")
    y_true_boundary = tf.sqrt(y_true_edge_x**2 + y_true_edge_y**2)

    # Compute boundaries for y_pred
    y_pred = tf.cast(y_pred, tf.float32)  # Ensure float type for convolution
    y_pred_edge_x = tf.nn.conv2d(y_pred, sobel_x, strides=[1, 1, 1, 1], padding="SAME")
    y_pred_edge_y = tf.nn.conv2d(y_pred, sobel_y, strides=[1, 1, 1, 1], padding="SAME")
    y_pred_boundary = tf.sqrt(y_pred_edge_x**2 + y_pred_edge_y**2)

    # Compute Dice loss between the boundary maps
    intersection = K.sum(y_true_boundary * y_pred_boundary)
    union = K.sum(y_true_boundary) + K.sum(y_pred_boundary)
    return 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)

@tf.keras.utils.register_keras_serializable()
def composite_loss(y_true, y_pred, alpha=0.7, beta=0.3, lambda1=0.5, lambda2=0.5):
    """
    Composite loss combining Tversky Loss and Boundary-Aware Loss.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param alpha: Tversky Loss parameter for false negatives.
    :param beta: Tversky Loss parameter for false positives.
    :param lambda1: Weight for Tversky Loss.
    :param lambda2: Weight for Boundary Loss.
    :return: Combined loss value.
    """
    loss_tversky = tversky_loss(y_true, y_pred, alpha, beta)
    loss_boundary = boundary_loss(y_true, y_pred)
    return lambda1 * loss_tversky + lambda2 * loss_boundary

class ElasticDeformation(ImageOnlyTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.elastic = ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)

    def apply(self, img, **params):
        return self.elastic(image=img)['image']


class PredictionCallback(Callback):
    def __init__(self, model_instance, sample_images, iteration_dir):
        """
        Callback to save predictions for multiple sample images after each epoch.

        :param model_instance: The NucleusSegmentationModel instance.
        :param sample_images: List of tuples [(original_image_1, processed_image_1), ...].
        :param iteration_dir: Directory to save predictions for this iteration.
        """
        super().__init__()
        self.model_instance = model_instance
        self.sample_images = sample_images  # Store as a list
        self.iteration_dir = iteration_dir

    def on_epoch_end(self, epoch, logs=None):
        # print(f"\nEpoch {epoch + 1}: Visualizing predictions for sample images...")
        for idx, (original_image, processed_image) in enumerate(self.sample_images):
            save_path = os.path.join(
                self.iteration_dir, f"epoch_{epoch + 1}_image_{idx + 1}.png"
            )
            self.model_instance.test_image(original_image, processed_image, save_path)


class WarmUpLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, target_lr, warmup_steps):
        """
        Warm-up learning rate schedule.

        :param initial_lr: Initial learning rate.
        :param target_lr: Target learning rate after warm-up.
        :param warmup_steps: Number of steps for warm-up.
        """
        super(WarmUpLearningRate, self).__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)  # Ensure float for TensorFlow ops

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Ensure `step` is a float Tensor
        warmup_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (step / self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: self.target_lr)

    def get_config(self):
        """
        Return the configuration of the learning rate schedule for serialization.
        """
        return {
            "initial_lr": self.initial_lr,
            "target_lr": self.target_lr,
            "warmup_steps": float(self.warmup_steps)  # Convert Tensor to float for serialization
        }

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of WarmUpLearningRate from a configuration dictionary.
        """
        return cls(**config)


class NucleusSegmentationModel:
    def __init__(self, input_size=(256, 256, 3), backbone='resnet34', initial_lr=1e-6, target_lr=1e-3, warmup_epochs=5):
        """
        Initialize the NucleusSegmentationModel with specified backbone and input size.

        :param input_size: Tuple specifying the input dimensions for the model.
        :param backbone: Backbone model to use in U-Net (default is 'resnet34').
        """
        self.input_size = input_size
        self.backbone = backbone
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = None
        self.preprocess_input = get_preprocessing(self.backbone)
        self.model = self.build_unet_model()

    def build_unet_model(self):
        """
        Build the U-Net model with warm-up learning rate.
        """
        # Define the warm-up learning rate schedule
        steps_per_epoch = self.steps_per_epoch or 100  # Set this dynamically in train_model
        total_warmup_steps = self.warmup_epochs * steps_per_epoch
        lr_schedule = WarmUpLearningRate(self.initial_lr, self.target_lr, total_warmup_steps)

        # Use the learning rate schedule in the optimizer
        optimizer = Adam(learning_rate=lr_schedule)

        # Build and compile the U-Net model
        model = Unet(
            backbone_name=self.backbone,
            input_shape=self.input_size,
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid'
        )
        model.compile(optimizer=optimizer, loss=composite_loss, metrics=['accuracy'])
        return model

    def save_model(self, model_save_path, train_images, val_images):
        """
        Save the trained model to a specified file path.

        :param val_images: validation images
        :param train_images: training images
        :param model_save_path: Path to save the model file.
        """
        #saving model
        self.model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Final epoch predictions and comparisons
        print("\nReloading model and comparing predictions...")
        reloaded_model = tf.keras.models.load_model(
            model_save_path,
            custom_objects={
                "composite_loss": composite_loss,
                "tversky_loss": tversky_loss,
                "boundary_loss": boundary_loss,
                "WarmUpLearningRate": WarmUpLearningRate
            }
        )

        # Directories for predictions
        save_dir = os.path.dirname(model_save_path)
        oblong_fitting_dir = os.path.join(save_dir, "final_predictions_oblong_fitting")
        os.makedirs(oblong_fitting_dir, exist_ok=True)

        # Combine training and validation images
        all_images = np.concatenate((train_images, val_images), axis=0)

        # Save predictions with oblong fitting applied
        for idx, image in enumerate(all_images):
            pred = reloaded_model.predict(np.expand_dims(image, axis=0))[0]
            pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            oblong_mask = self.fit_oblong_shape(pred_resized)
            mask_applied_image = self.blackout_background(image, oblong_mask)
            save_path = os.path.join(oblong_fitting_dir, f"final_oblong_pred_{idx + 1:03d}.tif")
            Image.fromarray(mask_applied_image).save(save_path)

        print(f"Final oblong-fitted predictions saved in {oblong_fitting_dir}.")

    @classmethod
    def load_model(cls, file_path, input_size=(256, 256, 3), backbone='resnet34'):
        """
        Load a saved model from a specified file path.

        :param file_path: Path to the saved model file.
        :param input_size: Input size of the model.
        :param backbone: Backbone used in the model.
        :return: An instance of NucleusSegmentationModel with the loaded model.
        """
        instance = cls(input_size=input_size, backbone=backbone)
        instance.model = keras_load_model(file_path, compile=False)
        instance.model.compile(optimizer=Adam(), loss=composite_loss, metrics=['accuracy'])
        print(f"Model loaded from {file_path}")
        return instance

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
        Preprocess an image from a file path or numpy array.

        :param image: File path to an image or a numpy array representing the image.
        :return: Preprocessed image array.
        """
        if isinstance(image, str):  # If input is a file path
            img_array = self.load_image(image)
        elif isinstance(image, np.ndarray):  # If input is already a numpy array
            img_array = image
        else:
            raise ValueError("Input must be a file path (str) or numpy array.")

        return self.preprocess_input(img_array)

    def fit_oblong_shape(self, mask):
        """
        Adjust the predicted mask to fit an oblong or irregular shape using convex hull and filling.
        :param mask: Predicted binary mask as a numpy array.
        :return: Adjusted binary mask.
        """
        if mask.max() <= 0:  # Return the mask immediately if it's empty
            return mask

        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.uint8)

        # Fill gaps and smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        adjusted_mask = np.zeros_like(mask)

        for contour in contours:
            if len(contour) > 5:  # Only process significant contours
                # Apply convex hull for irregular oblong fitting
                hull = cv2.convexHull(contour)
                cv2.drawContours(adjusted_mask, [hull], -1, 1, thickness=cv2.FILLED)

                # Fit an ellipse if the contour has enough points
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(adjusted_mask, ellipse, 1, -1)  # Fill the center oblong shape

        return adjusted_mask

    def load_images(self, image_folders, mask_folders, train_size=0.9):
        """
        Load and preprocess images and masks from multiple folders.

        :param image_folders: List of paths to folders containing brightfield images.
        :param mask_folders: List of paths to folders containing corresponding masks.
        :param train_size: Percentage of data to use for training within each folder.
        :return: Training and validation images and masks as numpy arrays.
        """
        train_images, train_masks = [], []
        val_images, val_masks = [], []
        if len(image_folders) != len(mask_folders):
            raise ValueError("Number of image folders and mask folders must be the same.")

        for image_folder, mask_folder in zip(image_folders, mask_folders):
            folder_images, folder_masks = [], []
            for img_name in os.listdir(image_folder):
                img_path = os.path.join(image_folder, img_name)
                mask_path = os.path.join(mask_folder, img_name)
                if not os.path.exists(mask_path):
                    print(f"Skipping {img_name}: No corresponding mask found in mask folder.")
                    continue
                img_array = self.preprocess_image(img_path)
                folder_images.append(img_array)
                with Image.open(mask_path) as mask:
                    mask = mask.convert("L")
                    mask = mask.resize(self.input_size[:2])
                    mask_array = img_to_array(mask) / 255.0
                    folder_masks.append(mask_array)

            folder_images = np.array(folder_images)
            folder_masks = np.array(folder_masks)

            # Split each folder's data into training and validation sets
            split_index = int(len(folder_images) * train_size)
            train_images.extend(folder_images[:split_index])
            train_masks.extend(folder_masks[:split_index])
            val_images.extend(folder_images[split_index:])
            val_masks.extend(folder_masks[split_index:])

        return np.array(train_images), np.array(train_masks), np.array(val_images), np.array(val_masks)

    def train_model(self, train_images, train_masks, val_images, val_masks,
                    epochs=50, batch_size=16, sample_images=None, save_dir=None,
                    iterations=1, callbacks=None):
        """
        Train the model with iterative feedback and optional callbacks.

        :param train_images: Training images.
        :param train_masks: Training masks.
        :param val_images: Validation images.
        :param val_masks: Validation masks.
        :param epochs: Number of epochs per iteration.
        :param batch_size: Batch size.
        :param sample_images: List of tuples [(original_image, processed_image), ...].
        :param save_dir: Directory to save predictions and model.
        :param iterations: Number of feedback iterations.
        :param callbacks: List of callbacks to apply during training.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.steps_per_epoch = len(train_images) // batch_size
        oblong_applied = False

        for iteration in range(iterations):
            print(f"\nStarting iteration {iteration + 1}/{iterations}...")
            iteration_dir = os.path.join(save_dir, f"iteration_{iteration + 1}")
            os.makedirs(iteration_dir, exist_ok=True)

            prediction_callbacks = []
            if sample_images:
                prediction_callbacks.append(PredictionCallback(self, sample_images, iteration_dir))

            all_callbacks = prediction_callbacks + (callbacks or [])

            # Augmentation pipeline
            elastic_deform = A.Compose([
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])

            def augment_image(img, mask):
                augmented = elastic_deform(image=img, mask=mask)
                return augmented['image'], augmented['mask']

            def generator(images, masks, batch_size):
                while True:
                    idx = np.random.choice(len(images), batch_size, replace=False)
                    batch_images = []
                    batch_masks = []
                    for i in idx:
                        img, mask = augment_image(images[i], masks[i])
                        batch_images.append(img)
                        batch_masks.append(mask)
                    yield np.array(batch_images), np.array(batch_masks)

            train_gen = generator(train_images, train_masks, batch_size)

            # Train the model
            history = self.model.fit(
                train_gen,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=(val_images, val_masks),
                epochs=epochs,
                callbacks=all_callbacks
            )

            # Check for accuracy > 90% and apply oblong-fitting post-processing
            final_epoch_metrics = history.history["accuracy"][-1]
            if final_epoch_metrics > 0.9 and not oblong_applied:
                print("Accuracy exceeded 90%. Applying oblong-fitting to predictions.")
                oblong_applied = True
                val_predictions = self.model.predict(val_images)
                val_masks = np.array([self.fit_oblong_shape(pred) for pred in val_predictions])

        return history

    def test_image(self, original_image, processed_image, save_path=None):
        """
        Predicts the mask for a single processed image, displays the original image
        alongside the predicted mask, and optionally saves the isolated nucleus image.

        :param original_image: The original, unprocessed image.
        :param processed_image: The preprocessed image to use for prediction.
        :param save_path: Optional; path where the final isolated nucleus image will be saved.
        """
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        predicted_mask = self.model.predict(processed_image)[0]  # Predict and remove batch dimension

        # Plot and optionally save the results
        self.plot_results(original_image, predicted_mask, save_path)

    def plot_results(self, original_image, predicted_mask, save_path=None):
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image = original_image.squeeze().astype(np.uint8)

        # Check if the mask is empty
        if predicted_mask.max() <= 0:
            print("Predicted mask is empty. Skipping adjustment.")
            adjusted_mask = predicted_mask
        else:
            # Adjust predicted mask only if it's not empty
            adjusted_mask = self.fit_oblong_shape(predicted_mask)

        # Fill the background using blackout logic
        isolated_nucleus = self.blackout_background(original_image, adjusted_mask)

        # Plot original image, predicted mask, and isolated nucleus
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image, cmap="gray" if len(original_image.shape) == 2 else None)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(adjusted_mask.squeeze(), cmap='gray')
        axes[1].set_title('Adjusted Predicted Mask')
        axes[1].axis('off')
        axes[2].imshow(isolated_nucleus)
        axes[2].set_title('Isolated Nucleus')
        axes[2].axis('off')

        if save_path is not None:
            isolated_nucleus_pil = Image.fromarray(isolated_nucleus)
            isolated_nucleus_pil.save(save_path)
            # print(f"Isolated nucleus image saved to {save_path}")

        plt.close(fig)

    def blackout_background(self, original_img, mask, threshold=0.5):
        """
        Blackout the background in the original image, keeping only the nucleus regions.

        :param original_img: The original image.
        :param mask: The predicted mask.
        :param threshold: Threshold to apply to the mask.
        :return: Image with the background blacked out.
        """
        # Ensure mask has proper dimensions
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # Remove unnecessary channel
        elif mask.ndim == 3 and mask.shape[-1] == 3:
            mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8 before applying cvtColor
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        elif mask.ndim != 2:
            raise ValueError(f"Unexpected mask dimensions: {mask.shape}")

        # Resize mask only if needed
        if mask.shape != original_img.shape[:2]:
            # print(f"Resizing mask from {mask.shape} to {original_img.shape[:2]}")
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply threshold to create binary mask
        mask = (mask > threshold).astype(np.uint8)

        # Prepare the image for masking
        original_rgb = (original_img.squeeze() * 255).astype(np.uint8)
        nucleus_only = original_rgb.copy()

        # Ensure the mask and image sizes are now the same
        if mask.shape != original_rgb.shape[:2]:
            raise ValueError(f"Mask size mismatch: Mask {mask.shape}, Image {original_rgb.shape[:2]}")

        # Blackout the background
        nucleus_only[mask == 0] = 0
        return nucleus_only

    def create_run_log(self, log_file_path, purpose, script_path):
        """
        Creates a log file detailing the parameters, purpose, and snapshot of the current code.

        :param log_file_path: Path to save the log file.
        :param purpose: Description of the purpose of the code.
        :param script_path: Path to the main script used to run the model.
        """
        log_content = ["### Purpose of the Run ###\n", purpose + "\n\n", "### Model Parameters ###\n",
                       f"Input Size: {self.input_size}\n", f"Backbone: {self.backbone}\n", "\n",
                       "### NucleusSegmentationModel Code ###\n"]

        # Add snapshot of this class
        model_code = inspect.getsource(NucleusSegmentationModel)
        log_content.append(model_code + "\n\n")

        # Add snapshot of the main script
        log_content.append("### Main Script Code ###\n")
        try:
            with open(script_path, 'r') as script_file:
                script_code = script_file.read()
                log_content.append(script_code + "\n")
        except Exception as e:
            log_content.append(f"Error loading script: {e}\n")

        # Write log to file
        with open(log_file_path, 'w') as log_file:
            log_file.writelines(log_content)

        print(f"Run log saved to {log_file_path}")
