# Smart Music Player

The Smart Music Player is a Webapp created using Python and Flask that recommends songs to a user based on their facial emotions.

A Convolutional Neural Network deep learning model created with Keras and trained using the "fer2013' dataset is being used in the app to detect the facial emotions of the user. Once the users emotions are detected, the app determines a genre most suitable for the user's emotions and recommends a song using the Spotipy API.

Currently the app is only displaying recommended songs, but a feature to connect to the user's spotify account and play the reccomended songs directly on Spotify is being worked on.