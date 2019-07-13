import logging
from progress.bar import IncrementalBar

def get_iterator(name, logger, loader):
    if logger.getEffectiveLevel() == logging.INFO:
        return IncrementalBar(name).iter(loader)
    else:
        return loader