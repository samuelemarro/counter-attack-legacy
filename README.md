[![Build Status](https://travis-ci.com/samuelemarro/counter-attack.png?branch=master)](https://travis-ci.com/samuelemarro/anti-attacks)
# Counter-Attack

Counter-Attack is an adversarial attack-based defense against adversarial attacks. When we receive a potentially adversarial sample, we use adversarial attacks to estimate the distance to the decision boundary. If the distance is below a certain threshold, we reject the sample.

# Troubleshooting

**I have increased the batch size, but the speed is the same**

Check that parallelization is enabled and increase --attack-workers as well.

**Some adversarial samples are actually classified correctly**

Depending on the approximations used by CUDA, some adversarial samples close to the decision boundary might be classified correctly. To prevent this, use --no-cuda. Keep in mind that these samples are so close to the decision boundary (usually around 1e-5 L-inf distance) that they might trick the same network in a different context.

# Acknowledgements

We would like to thank Wei Yang and all the contributors of [pytorch_classification](https://github.com/bearpaw/pytorch-classification) for their pretrained CIFAR models.
