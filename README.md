# Face ID
## Created a custom facial recognition model

Created a Siamese Convolutional Nerual Network for one shot learning of facial features. Implemented from this paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

Idea behind it is training two embedding networks with the same weights. They embedd two images in tandem and then a distance layer takes both embeddings and determines if they are the same class or not i.e. the same person.

Currently, in `main.py` I have a streamlit dashboard that can be run by `streamlit run main.py`. This will let you upload an image, then automatically detect faces and run the model on a few baseline images of myself to determine if I am in these pictures. 

## steps to improve
1) Repo could be a bit more organized, add a reqs.txt, break down code a little and possibly take the training pipeline out of a notebook
2) Need extensively more data. Right now, the data I have is unstructured, so I will need to go in and label most of it.
