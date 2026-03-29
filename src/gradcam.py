from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

try:
    import tensorflow as _tf  # type: ignore
except Exception:  # pragma: no cover
    _tf = None  # type: ignore


def get_last_conv_layer(model: "tf.keras.Model") -> "tf.keras.layers.Layer":
    """Find the last convolution-like layer in a Keras model.

    Raises:
        ValueError: If no suitable conv layer is found.
    """
    if _tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM.")

    conv_types = (
        _tf.keras.layers.Conv2D,
        _tf.keras.layers.SeparableConv2D,
        _tf.keras.layers.DepthwiseConv2D,
        _tf.keras.layers.Conv2DTranspose,
    )

    def _rank_of_layer_output(layer: "tf.keras.layers.Layer") -> int | None:
        try:
            out = layer.output
        except Exception:
            return None
        shape = getattr(out, "shape", None)
        if shape is None:
            return None
        r = getattr(shape, "rank", None)
        if r is not None:
            return int(r)
        try:
            return len(shape)
        except Exception:
            return None

    # Prefer top-level layers: nested submodel internals are often not reachable
    # from the outer graph (e.g., Sequential(MobileNetV2(...), ...)).
    for layer in reversed(model.layers):
        rank = _rank_of_layer_output(layer)
        if isinstance(layer, conv_types) and rank == 4:
            return layer

    # Fallback: last top-level feature map (B, H, W, C).
    for layer in reversed(model.layers):
        rank = _rank_of_layer_output(layer)
        if rank == 4:
            return layer

    raise ValueError("No suitable 4D feature layer found in model; cannot compute Grad-CAM.")


def _pick_score_tensor(preds: "tf.Tensor", class_index: int | None) -> "tf.Tensor":
    """Select the scalar score to explain.

    Supports:
    - Binary sigmoid output (B, 1)
    - Softmax/logits output (B, C)
    """
    preds = _tf.convert_to_tensor(preds)

    if preds.shape.rank != 2:
        raise ValueError(f"Unsupported model output rank: {preds.shape}")

    n_classes = int(preds.shape[-1]) if preds.shape[-1] is not None else None

    if n_classes == 1:
        p = preds[:, 0]
        if class_index is None:
            score = _tf.where(p >= 0.5, p, 1.0 - p)
        else:
            if class_index not in (0, 1):
                raise ValueError("For sigmoid output, class_index must be 0 or 1.")
            score = p if class_index == 1 else (1.0 - p)
        return score

    if n_classes is None:
        raise ValueError("Model output last dimension is unknown; cannot pick class score.")

    if class_index is None:
        class_index = int(_tf.argmax(preds[0]).numpy())

    if class_index < 0 or class_index >= n_classes:
        raise ValueError(f"class_index {class_index} out of range for {n_classes} classes")

    return preds[:, class_index]


def generate_gradcam(
    *,
    model: "tf.keras.Model",
    input_image: np.ndarray,
    class_index: int | None = None,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: Keras model.
        input_image: Image array shaped (H, W, C) or (1, H, W, C), float32 recommended.
        class_index: Which class to explain. For sigmoid models: 0 (FAKE) or 1 (REAL).
            If None, uses the predicted class (binary) or argmax (multi-class).
        conv_layer_name: Optional layer name override. If None, uses last conv layer.

    Returns:
        heatmap: float32 array (H, W) normalized to [0, 1].
    """
    if _tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM.")

    x = np.asarray(input_image)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"input_image must be (H,W,C) or (1,H,W,C). Got {x.shape}.")

    x_tf = _tf.convert_to_tensor(x, dtype=_tf.float32)
    h, w = int(x_tf.shape[1]), int(x_tf.shape[2])

    if conv_layer_name is not None:
        conv_layer = model.get_layer(conv_layer_name)
    else:
        conv_layer = get_last_conv_layer(model)

    def _build_grad_model() -> "tf.keras.Model":
        if isinstance(model, _tf.keras.Sequential):
            inp_shape = tuple(int(d) if d is not None else None for d in model.inputs[0].shape[1:])
            inputs = _tf.keras.Input(shape=inp_shape)
            t = inputs
            conv_out = None
            for layer in model.layers:
                if isinstance(layer, _tf.keras.layers.InputLayer):
                    continue
                t = layer(t)
                if layer is conv_layer:
                    conv_out = t
            if conv_out is None:
                raise RuntimeError("Could not capture conv layer output while rebuilding Sequential graph.")
            return _tf.keras.Model(inputs=inputs, outputs=[conv_out, t])

        return _tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.outputs[0]])

    grad_model = _build_grad_model()

    with _tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x_tf, training=False)
        score_vec = _pick_score_tensor(preds, class_index)
        score = _tf.reduce_sum(score_vec)

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute Grad-CAM gradients (got None).")

    # Global average pooling over spatial dims
    pooled_grads = _tf.reduce_mean(grads, axis=(1, 2))  # (1, channels)

    conv_outputs = conv_outputs[0]  # (h', w', channels)
    weights = pooled_grads[0]  # (channels,)

    cam = _tf.reduce_sum(conv_outputs * weights, axis=-1)  # (h', w')
    cam = _tf.nn.relu(cam)

    cam = _tf.image.resize(cam[..., _tf.newaxis], (h, w), method="bilinear")[..., 0]

    cam_min = _tf.reduce_min(cam)
    cam_max = _tf.reduce_max(cam)
    denom = _tf.where(cam_max - cam_min > 1e-8, cam_max - cam_min, _tf.constant(1.0, dtype=cam.dtype))
    cam_norm = (cam - cam_min) / denom

    return cam_norm.numpy().astype(np.float32)


def overlay_heatmap(
    *,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """Colorize a heatmap and blend it over an RGB image.

    Args:
        image_rgb: (H, W, 3) float in [0,1] or uint8 in [0,255].
        heatmap: (H, W) float in [0,1].
        alpha: Blend amount for heatmap.
        colormap: Matplotlib colormap name (default: jet).

    Returns:
        heatmap_rgb_uint8: (H, W, 3) uint8
        overlay_uint8: (H, W, 3) uint8
    """
    import matplotlib

    img = np.asarray(image_rgb)
    hm = np.asarray(heatmap, dtype=np.float32)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image_rgb must be (H,W,3). Got {img.shape}.")
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be (H,W). Got {hm.shape}.")

    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    img_f = np.clip(img_f, 0.0, 1.0)
    hm = np.clip(hm, 0.0, 1.0)

    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps.get_cmap(colormap)
    else:  # pragma: no cover
        cmap = matplotlib.cm.get_cmap(colormap)

    hm_rgb = cmap(hm)[..., :3].astype(np.float32)

    a = float(alpha)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    overlay = (1.0 - a) * img_f + a * hm_rgb

    hm_uint8 = (hm_rgb * 255.0).round().astype(np.uint8)
    overlay_uint8 = (np.clip(overlay, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    return hm_uint8, overlay_uint8
