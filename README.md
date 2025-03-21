# Negative Rejection Steering
NRS seeks to replace the 'naive' linear interpolation of Classifier Free Guidance with a more nuanced steering of the generation process.

This is accomplised in 3 steps:
1. **Displacement**: The conditioned output tensor is displaced in the direction of the rejection of the unconditioned tensor on the conditioned tensor. This lengthens the tensor in a direction perpendicular to it's direction without affecting the positive guidance. The tensor is displaced by the rejection x the Displacement parameter.
2. **Squashing**: The displaced tensor is rescaled towards the original length of the conditioned tensor. This means for high displacement scaling values the tensor 'turns' away from the unconditioned direction, which for very negative displacements, it turns towards the unconditioned tensor. 0 displacement outputs the original conditioned tensor.
3. **Stretching**: The post-squash 'steered' tensor is stretched towards the direction of the original conditioned tensor. The more sharp the steering the less pronounced the stretch is, with fully aligned tensors being stretched the full stretch scale parameter. 1x stretch adds 100% length to the tensor.

# Examples (TODO)
