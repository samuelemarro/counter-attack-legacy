
from counter_attack.cli.commands import \
    accuracy, approximation_dataset, attack, \
    check, defense, detector_roc, \
    train_approximator, train_model

commands = [
    accuracy.accuracy,
    approximation_dataset.approximation_dataset,
    attack.attack,
    check.check,
    defense.defense,
    detector_roc.detector_roc,
    train_approximator.train_approximator,
    train_model.train_model
]