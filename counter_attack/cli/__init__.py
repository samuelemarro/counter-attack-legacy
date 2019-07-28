
from counter_attack.cli.commands import \
    accuracy, approximation_dataset, attack, \
    boundary_distance, \
    check, defense, detector_roc, \
    radius, \
    train_approximator, train_model

commands = [
    accuracy.accuracy,
    approximation_dataset.approximation_dataset,
    attack.attack,
    boundary_distance.boundary_distance,
    check.check,
    defense.defense,
    detector_roc.detector_roc,
    radius.radius,
    train_approximator.train_approximator,
    train_model.train_model
]