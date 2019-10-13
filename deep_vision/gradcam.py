import numpy as np
import keras.backend as K
import cv2


def gradcam(actual_image, normalised_image, model, layer, n_channels):
    '''
    returns grad cam activations for the normalised image
    parameters:
    actual_image: unnormalised image
    normalised_image: normalised actual image
    model: CNN model used
    layer: layer in the model from which to compute gradcam activations
    n_channels: number of channels in the output feature map of layer
    '''

    x = np.expand_dims(normalised_image, axis=0)
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer)

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for j in range(n_channels):
        conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (normalised_image.shape[1], normalised_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(actual_image, 0.6, heatmap, 0.4, 0)
    return superimposed_img
