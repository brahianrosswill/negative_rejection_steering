# Negative Rejection Steering
NRS seeks to replace the 'naive' linear interpolation of Classifier Free Guidance with a more nuanced and composable steering of the generation process with better mathematical basis.

### Math Demonstration
![image](https://github.com/user-attachments/assets/01fabaff-8499-45f6-adad-d54b2c2fb7f1)
[Interactive Graph on Math3D.org](https://www.math3d.org/aTJW4UZtCh)

This is accomplised in 3 steps:
1. **Displacement**: The conditioned output tensor is displaced in the direction of the rejection of the unconditioned tensor on the conditioned tensor. This lengthens the tensor in a direction perpendicular to it's direction without affecting the positive guidance. The tensor is displaced by the rejection x the Displacement parameter.
2. **Squashing**: The displaced tensor is rescaled towards the original length of the conditioned tensor. This means for high displacement scaling values the tensor 'turns' away from the unconditioned direction, which for very negative displacements, it turns towards the unconditioned tensor. 0 displacement outputs the original conditioned tensor.
3. **Stretching**: The post-squash 'steered' tensor is stretched towards the direction of the original conditioned tensor. The more sharp the steering the less pronounced the stretch is, with fully aligned tensors being stretched the full stretch scale parameter. 1x stretch adds 100% length to the tensor.

# Alpha Release
Implements NRS with Skew, Stretch, and Squash parameters.

## Parameters
Skew and Stretch are roughly similar to CFG, but decomposed, with `Stretch + Skew = 2 * CFG`, roughly.

**Skew** changes the 'direction' of generation, which should result in changes to the content and composition of the image.

**Stretch** changes to 'amplification' of generation, which should result in stronger prompt representation.

**Squash** 'normalizes' the resulting guidance back towards the original amplitude with 1.0 being the same amplitude, while 0.0 is the unmodified amplitude resulting from the Squash and Stretch functions.

## Beginner How-To
1. Set Squash to 0.0
2. Set Skew & Stretch each to your normal CFG Scale setting
3. Test some generation. Results should be 'similar' in quality to CFG
4. Adjust Skew up/down to change content and composition
5. Adjust Stretch up/down to change strength of image aspects and colors
6. Adjust Squash up to remove artifacts and color burn (these will tend to be replaced by additional or extraneous details and elements)

**Tip**: You can experiment with negative values for Skew and Stretch as well to see what the model 'believes' your negative prompt 'means'.
