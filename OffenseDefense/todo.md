* Implement a comparison with the distribution of distances
* Modularize
* Fix Matplotlib not responding
* Repeat basic_test with increased iterations
* Check FineTuningAttack
* Introduce more assertions and unit tests
* PersistentAttack (tries the same attack with different epsilons until it succeeds)
* Find a policy for failures in adversarial attacks
* Check custom filters for utils.Filter
* Check get_best_threshold
* Add support for test recipes? (such as files with test specifications)
* Implement a non-parallel version of the attacks (for comparison with the parallel version and testing)
* Handling parallel attacks requires quite a lot of considerations. Hide them from the user
* Implement logging
* The RandomDirectionAttack doesn't return a correct Adversarial (e.g. wrong distance) if unpack=False
* batch_attack.get_anti_adversarials is a bit a repetition of batch_attack.get_adversarials
* get_adversarials and get_anti_adversarials returning a Filter is confusing
* Checking the correct classifications in get_adversarial_samples is sometimes redundant