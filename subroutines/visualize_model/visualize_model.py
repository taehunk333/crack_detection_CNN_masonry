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

from contextlib import redirect_stdout

def model_summary(model, folder_summary):
    """
    save the summary to txt
    """
    
    with open(folder_summary, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
def visualize_model(model, folder_plot, folder_summary):
    """ 
    from contextlib import redirect_stdout
    save the summary to txt,
    plot the model to png
    """
    
    from tensorflow.keras.utils import plot_model
    
    plot_model(model, to_file=folder_plot, show_shapes=True)
    model_summary(model, folder_summary)


def visualize_model_tf(model, folder_plot, folder_summary):
    """ 
    for models built with tf.keras use this function to visualize them
    from contextlib import redirect_stdout
    save the summary to txt,
    plot the model to png
    """
    
    import tensorflow as tf
  
    tf.keras.utils.plot_model(model, folder_plot, show_shapes=True)
    model_summary(model, folder_summary)
                
