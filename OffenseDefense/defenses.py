import art
import foolbox
import numpy as np
import torch

from . import batch_attack, detectors


class PreprocessorDefenseModel(foolbox.models.ModelWrapper):
    def __init__(self, foolbox_model, preprocessor):
        super().__init__(foolbox_model)
        self.preprocessor = preprocessor

    def _preprocess(self, image):
        batch = np.array([image])
        preprocessed_batch = self.preprocessor(batch, None)
        return preprocessed_batch[0]

    def predictions(self, image):
        image = self._preprocess(image)

        return self.wrapped_model.predictions(image)

    def batch_predictions(self, images):
        images = [self._preprocess(image) for image in images]
        images = np.array(images)

        return self.wrapped_model.batch_predictions(images)
