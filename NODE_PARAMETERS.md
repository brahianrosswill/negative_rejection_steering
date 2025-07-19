# Node Parameters Documentation

This document provides a detailed explanation of the custom nodes found in `nodes_NRS.py`. These nodes are designed for advanced control over the image generation process in ComfyUI.

## `NRS` Node

The `NRS` node implements Negative Rejection Sampling for v-prediction diffusion models. It works by modifying the conditional and unconditional predictions to steer the generated image more effectively.

### Parameters

*   **`model`**: The input diffusion model. This should be a v-prediction model for this node to work correctly.
*   **`version`**: This parameter selects the specific NRS algorithm to use. Different versions implement different mathematical formulas for calculating the final prediction, leading to varied results. It's recommended to experiment with different versions to find the one that best suits your needs.
    *   **Available Versions**: `v1`, `v2`, `v3`, `v4`, `v0.4.1`, `v0.4.2`, `v0.4.3`, `v0.4.4`, `v0.4.5`
*   **`skew`**: Controls the strength of the rejection of the unconditional prompt.
    *   **Effect**: Higher values steer the image further away from the unconditional (negative) prompt, leading to more creative and less generic outputs.
    *   **Practical Use**:
        *   `0-2`: Subtle effect, close to standard generation.
        *   `2-6`: Good for most use cases, balances creativity and prompt adherence.
        *   `>6`: Can lead to stylized, abstract, or artifact-heavy images.
*   **`stretch`**: Amplifies details that are unique to the conditional (positive) prompt.
    *   **Effect**: Enhances sharpness and definition of features in the image.
    *   **Practical Use**:
        *   `0-1`: Minimal detail enhancement.
        *   `1-3`: Good for adding sharpness and clarity.
        *   `>3`: May cause over-sharpening and halos.
*   **`squash`**: Constrains the magnitude of the final prediction to be closer to the original conditional prediction.
    *   **Effect**: Prevents the image from becoming overly saturated or having "blown out" highlights.
    *   **Practical Use**:
        *   `0.0`: No squashing, can result in high contrast and saturation.
        *   `1.0`: Full squashing, maintains a more "natural" look.
        *   `0.0-1.0`: Allows for fine-tuning the balance.
*   **`normalize_strength`**: Controls the intensity of Adaptive Normalization.
    *   **Effect**: Adjusts the color and contrast of the output to match the conditional prediction.
    *   **Practical Use**:
        *   `0.0`: No normalization.
        *   `1.0`: Full normalization, can fix color casts and improve tonal balance.
*   **`adaptive_k`**: A parameter for Adaptive Normalization that limits the change in standard deviation.
    *   **Effect**: Prevents extreme shifts in the image's dynamic range.
    *   **Practical Use**: The default value of `10.0` is usually sufficient.
*   **`epsilon`**: A small value to prevent division-by-zero errors.
    *   **Effect**: No noticeable effect on the final image.
    *   **Practical Use**: Leave at its default value.

## `NRSFDG` Node

The `NRSFDG` node combines Negative Rejection Sampling with Frequency Dependent Guidance (FDG). This allows for applying different levels of guidance to different frequency bands of the image, giving you more nuanced control over the final output. This node is for v-prediction models.

### Parameters

This node includes all the parameters from the `NRS` node, plus additional parameters for controlling the FDG.

*   **`model`**: See the `NRS` node documentation.
*   **`version`**: See the `NRS` node documentation.
*   **`skew`**: See the `NRS` node documentation.
*   **`stretch`**: See the `NRS` node documentation.
*   **`squash`**: See the `NRS` node documentation.
*   **`normalize_strength`**: See the `NRS` node documentation.
*   **`adaptive_k`**: See the `NRS` node documentation.
*   **`epsilon`**: See the `NRS` node documentation.

#### FDG Parameters

*   **`guidance_scale_high`**: The guidance scale (CFG) for high-frequency details.
    *   **Effect**: Higher values lead to sharper and more pronounced fine details.
    *   **Practical Use**: Useful for enhancing textures and small features.
*   **`guidance_scale_low`**: The guidance scale (CFG) for low-frequency details.
    *   **Effect**: Higher values result in a more coherent and stable overall composition.
    *   **Practical Use**: Useful for ensuring the main subject and structure of the image are well-defined.
*   **`levels`**: The number of levels in the Laplacian pyramid.
    *   **Effect**: More levels provide finer control over frequency bands, but increase computation time.
    *   **Practical Use**: `2-4` levels are a good starting point.
*   **`fdg_steps`**: The number of diffusion steps to apply FDG.
    *   **Effect**: Applying FDG for fewer steps can speed up the generation process.
    *   **Practical Use**: A lower value can be a good trade-off between speed and quality.

## Practical Examples and Use Cases

### Achieving a "Cinematic" Look

For a cinematic look with sharp details and good composition, you can use the `NRSFDG` node with the following settings:

*   **`version`**: `v0.4.5` (or experiment to find your preference)
*   **`skew`**: `3.0`
*   **`stretch`**: `1.5`
*   **`squash`**: `0.5`
*   **`normalize_strength`**: `0.8`
*   **`guidance_scale_high`**: `10.0`
*   **`guidance_scale_low`**: `4.0`
*   **`levels`**: `3`
*   **`fdg_steps`**: `10`

This combination will give you strong guidance on the details, while allowing for some creative freedom in the overall composition. The `normalize_strength` will help to maintain a balanced color palette.

### Creating Stylized and Artistic Images

To create more stylized or abstract images, you can push the `skew` and `stretch` parameters higher in the `NRS` or `NRSEpsilon` nodes:

*   **`version`**: `v2`
*   **`skew`**: `8.0`
*   **`stretch`**: `4.0`
*   **`squash`**: `0.2`
*   **`normalize_strength`**: `0.5`

These settings will create a highly distorted and artistic effect. Be prepared for unexpected and potentially chaotic results.

### Correcting Color Casts

If you are getting images with a noticeable color cast, you can use the `normalize_strength` parameter in any of the nodes to correct it.

*   **`normalize_strength`**: `1.0`

This will force the output image to adopt the color statistics of the conditional prediction, which can often mitigate color balance issues.

## `NRSEpsilon` Node

The `NRSEpsilon` node is the counterpart to the `NRS` node for epsilon-prediction diffusion models. It applies the same Negative Rejection Sampling techniques, but is specifically designed to work with models that predict the noise (epsilon) rather than the v-parameter.

### Parameters

The parameters for the `NRSEpsilon` node are identical to the `NRS` node. The key difference is that this node should be used with epsilon-prediction models.

*   **`model`**: The input diffusion model. This should be an epsilon-prediction model.
*   **`version`**: See the `NRS` node documentation.
*   **`skew`**: See the `NRS` node documentation.
*   **`stretch`**: See the `NRS` node documentation.
*   **`squash`**: See the `NRS` node documentation.
*   **`normalize_strength`**: See the `NRS` node documentation.
*   **`adaptive_k`**: See the `NRS` node documentation.
*   **`epsilon`**: See the `NRS` node documentation.
