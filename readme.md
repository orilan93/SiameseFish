# SiameseFish

SiameseFish is a classifier of individuals of the _Symphodus melops_ species.
It uses deep metric learning to create embeddings from images and predicts the individual
from this using nearest neighbor. A high level documentation of the system can be found [here](documentation.md).

## Setup

1. Create a virtual environment (recommended).

    ```python -m venv venv```

2. Activate the environment.

    On macOS and Linux:

    ```source env/bin/activate```

    On Windows:

    ```.\env\Scripts\activate```

3. Install dependencies

    ```pip install -r requirements.txt```

## Configure

A brief guidance on how to configure the system.

1. Configure and run data_extraction.py to extract the images that are useful from the drive.
2. Configure and run preprocess_data.py to preprocess the images ready for training.
3. Configure and run embeddings.py to train the embedding network.
4. Configure and run distance_classifier.py to train and evaluate the classifier.

## Experiments


## Modules

Main modules:

- config.py contains different configurations.
- data.py contains functions for retrieving and manipulating data.
- data_extraction.py extracts data from the drive.
- direction_labeler.py is a tool for labeling the direction of the fish in an efficient manner.
- distance_classifier.py is classifies embeddings to their nearest neighbors in the support set.
- embeddings.py trains up an embedding model.
- metrics.py contains metrics and loss functions.
- models.py contains all the models used by the system.
- preprocess_data.py preprocesses the data and puts it into a structure ready for training.
- utils.py contains various support functions that are often needed.

Other modules:

- classifiction.py contains the different classifiers and is made to easily switch between them.
- cnn_classifier.py is a simple cnn classifier.
- direction_classifier.py classifies the direction the fish is facing.
- embeddings_siamese.py embedding learner using siamese network.
- embeddings_ohnm.py embedding learner using online hard negative mining.
- embeddings_triplet.py embedding learner using offline triplet siamese network.
- main.py acts as the main inference script to classify images.
- mlp_classifier.py classifies embeddings using mlp.
- playground.py exists to try out small snippets.
- region.py predicts a certain region of the fish in an image.
- saliency.py extracts the pattern of a fish in the form of a binary image.
- testing.py contains various tests.
- visualize_augmentations.py shows samples from the augmented dataset.