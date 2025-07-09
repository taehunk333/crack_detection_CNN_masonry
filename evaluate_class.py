"""
The following code was produced for the Journal paper 
"Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning"
by D. Dais, İ. E. Bal, E. Smyrou, and V. Sarhosis published in "Automation in Construction"
in order to apply Deep Learning and Computer Vision with Python for crack detection on masonry surfaces.

In case you use or find interesting our work please cite the following Journal publication:

D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces 
using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. 
https://doi.org/10.1016/j.autcon.2021.103606.

@article{Dais2021,
author = {Dais, Dimitris and Bal, İhsan Engin and Smyrou, Eleni and Sarhosis, Vasilis},
doi = {10.1016/j.autcon.2021.103606},
journal = {Automation in Construction},
pages = {103606},
title = {{Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning}},
url = {https://linkinghub.elsevier.com/retrieve/pii/S0926580521000571},
volume = {125},
year = {2021}
}

The paper can be downloaded from the following links:
https://doi.org/10.1016/j.autcon.2021.103606
https://www.researchgate.net/publication/349645935_Automatic_crack_classification_and_segmentation_on_masonry_surfaces_using_convolutional_neural_networks_and_transfer_learning/stats

The code used for the publication can be found in the GitHb Repository:
https://github.com/dimitrisdais/crack_detection_CNN_masonry

Author and Moderator of the Repository: Dimitris Dais

For further information please follow me in the below links
LinkedIn: https://www.linkedin.com/in/dimitris-dais/
Email: d.dais@pl.hanze.nl
ResearchGate: https://www.researchgate.net/profile/Dimitris_Dais2
Research Group Page: https://www.linkedin.com/company/earthquake-resistant-structures-promising-groningen
YouTube Channel: https://www.youtube.com/channel/UCuSdAarhISVQzV2GhxaErsg  

Your feedback is welcome. Feel free to reach out to explore any options for collaboration.
"""

import os
import sys
from tensorflow.keras.models import model_from_json

# Import custom losses and metrics
from subroutines.loss_metric.loss_metric import (
    Weighted_Cross_Entropy,
    Focal_Loss,
    F1_score_Loss,
    F1_score_Loss_dil,
    Recall,
    Precision,
    Precision_dil,
    F1_score,
    F1_score_dil
)

class LoadModel:
    def __init__(self, args, IMAGE_DIMS, BS):
        self.args = args
        self.IMAGE_DIMS = IMAGE_DIMS
        self.BS = BS

    def load_pretrained_model(self):
        """
        Load a pretrained model by building the architecture in code and loading weights.
        JSON loading is NOT supported for models with custom losses/metrics.
        """

        model_name = self.args["model"]
        networks_path = os.path.join(self.args["main"], 'networks')
        sys.path.append(networks_path)

        if model_name == 'Unet':
            from Unet import Unet  # Capital U matches Unet.py

            model = Unet(
                IMAGE_DIMS=(self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]),
                n_filters=self.args.get('N_FILTERS', 64),
                kernel_initializer=self.args.get('init', 'he_normal'),
                dropout=self.args.get('dropout'),
                batchnorm=self.args.get('batchnorm', True),
                regularization=self.args.get('regularization')
            )


            weights_path = os.path.join(self.args['weights'], self.args['pretrained_filename'])
            model.load_weights(weights_path)

        elif model_name == 'DeepCrack':
            from edeepcrack_cls import Deepcrack

            model = Deepcrack(input_shape=(self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))
            weights_path = os.path.join(self.args['weights'], self.args['pretrained_filename'])
            model.load_weights(weights_path)

        elif model_name == 'Deeplabv3':
            from model import Deeplabv3

            model = Deeplabv3(input_shape=(self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))
            weights_path = os.path.join(self.args['weights'], self.args['pretrained_filename'])
            model.load_weights(weights_path)

        elif model_name == 'MyCrackNet':
            from mycracknet import MyCrackNet

            model = MyCrackNet(input_shape=(self.BS, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))
            weights_path = os.path.join(self.args['weights'], self.args['pretrained_filename'])
            model.load_weights(weights_path)

        elif self.args.get('save_model_weights') == 'model':
            raise ValueError(
                "Loading full model from JSON is not supported in 'evaluation' mode.\n"
                "Use weights loading mode instead by setting args['save_model_weights'] == 'weights'."
            )

        else:
            raise NotImplementedError(f"Model loading not supported for model '{model_name}'.")

        return model