default_architecture_names = {
    'cifar10': 'densenet (depth=100, growth_rate=12)',
    'cifar100': 'densenet (depth=100, growth_rate=12)',
    'imagenet': 'densenet (depth=161, growth_rate=48)'
}

datasets = ['cifar10', 'cifar100', 'imagenet']
differentiable_attacks = ['bim', 'deepfool', 'fgsm', 'random_pgd']
black_box_attacks = ['boundary']
supported_attacks = differentiable_attacks + black_box_attacks

parallelizable_attacks = ['deepfool', 'fgsm']

supported_distance_tools = ['counter-attack']
cache_distance_tools = ['counter-attack']
supported_standard_detectors = []

supported_detectors = supported_distance_tools + supported_standard_detectors

supported_preprocessors = ['feature-squeezing', 'spatial-smoothing']

supported_ps = ['2', 'inf']
