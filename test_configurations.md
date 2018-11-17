Remember to run the following combinations of tests/parameters:
* Tests:
  * Visual: just checking that images are properly loaded
  * Basic: Estimate the distance of genuine and adversarial samples. Check that genuines >> adversarials
  * Approximation: Compare the anti-attack estimates with the random direction estimates. Check that anti-attack < random direction
* Settings: {genuine, adversarial, random}
* Distance metric: {2, Infinity}