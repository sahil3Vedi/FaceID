Facial Recognition algorithm that differentiates the faces of 5 prominent world leaders and myself using the face_recogntion.py library.
This library uses a set of pre_trained weights to determine facial encodings of a given image. The encodings can be compared among one another using either Euclidean distance or similarity measures to find the closest fit.

# Algorithm:
1. Collect Training Data: Ideally a single well augmented image for a person is enough to obtain their face encodings. We can have multiple images of a single person but face_recogntion.py has well optimised weights for the task the obtained encodings will be fairly unique for each individual with little to no margin of repetition.
2. Training: Obtain a list of known faces and encodings from the image.
3. Testing: Obtain encodings from a previously unkown image. We match the encodings to the list of known (trained) encodings and determine the encoding with the closest Euclidean distance. The name matching with the selected encoding is the response of the model.

# Using the model
Download the repository, open terminal and navigate to the repository's directory.
>python3 train.py
