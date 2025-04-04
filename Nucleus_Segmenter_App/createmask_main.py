from tensorflow.python.eager.context import get_config

from CreateMask import CreateMask
import tensorflow as tf
import tensorflow.keras.backend as K

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


if __name__ == '__main__':
    model_path = r"C:\Users\jason\PycharmProjects\nucleus_outline\unet\models\rn34\uint16\RN34_NSM_hela_jpg_V1.keras"
    mask_output_path = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-04-03_HelaS3_H3-2-Halo_FA\masks"
    brightfield_images = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-04-03_HelaS3_H3-2-Halo_FA\BF_images\noTreatment"
    cleaned_movies = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-04-03_HelaS3_H3-2-Halo_FA\2025-04-03_HeLaS3_H3-2-Halo_c25_noTreatment\cleaned_movies"

    bf_images_to_movies_index_formula = "int(index / 2)"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "composite_loss": composite_loss,
            "tversky_loss": tversky_loss,
            "boundary_loss": boundary_loss,
            "WarmUpLearningRate": WarmUpLearningRate
        }
    )

    mask_creator = CreateMask(model, mask_save_folder=mask_output_path)
    mask_creator.create_mask(brightfield_images, cleaned_movies, bf_images_to_movies_index_formula, verify=True)
