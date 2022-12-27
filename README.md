# Face ID - Create a custom facial recognition model

## What is it?

For this project, I created a Siamese Convolutional Nerual Network for one shot learning of facial features. Implemented the model in accordance to this paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## How it works

The idea behind it is simple. I trained two embedding networks with the same weights that are connected by a distance layer. 
In tandem, they embedd two images using a 6 layer CNN. Then a distance layer takes both embeddings and determines if they are the same class or not i.e. the same person. If the two pictures are similar, the model will return 1, otherwise it will return 0.

This model was trained by myself, on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset collected by umass Amherst, and a set of pictures of myself that I collected. The data to train this model, must be in the form of 2 photos and a label of 1 or 0 to inform the model if these are the same face or not. A positive or a negative photo is paired up with an anchor photo. An anchor photo is what the model compares the other picture to. For the purpose of our data set, a positive photo is a picture of myself while a negative photo is taken from the external data set. I have 500 positive, and 500 negative examples.



## How to use it
Currently, in `main.py` I have a streamlit dashboard that can be run by `streamlit run main.py`. This dashboard allows you to upload an image. It will use use an OpenCV facial segementation tool to extract all the faces in the picture. Then, the dashboard will highlight the face with a green square if that is myself, Omar Shoura. 

Since this is a oneshot classificiation model, it must compare the given faces to another face. In this case, I compare it to 50 pictures stored locally.

To run this streamlit, you must have these in your directory:
- folder called verification_data with the provided pictures of myself (in repo)
- `siamese_model.h5` file which contains the model itself


## What can be improved
There is a lot I can do to improve this model and the experience as a whole. Here is an unordered list of what I would like to work on next
1) Create a proper `requirements.txt` file for the streamlit dashboard. That would make usage of my app a lot easier
2) Train the model outside of a notebook and wrap it in an mlflow model or something of the sorts. My current method of training, tuning, iterating, and saving the model is a bit disorganized and I can fix that by having a more robust process using mlflow for example
3) Train the model to recognize just myself. The way I set up this model, it compares two pictures and when I use it, I simply ask it to compare the given picture to a stored picture of myself. While that may be more versatile, it burdensome for my usecase where I am only interested in it recognizing myself. Instead of twin embeddings, I would have a single embedding architecture and feed it pictures of myself with positive labels.
4) I also believe I can do a lot to improve the quality of my data. All the training data of myself is from a very limited period of time (only 10 or so minutes), and this seriously limits the models ability to extrapolate. I want to use all the photos I have of myself, it would just take a lot of time to label them all. If I can't do that though, I would like to create some artifical data/ add noise to my existing dataset using some pretty standard techniques. That would certainly boost the performance of my model in new scenarios.