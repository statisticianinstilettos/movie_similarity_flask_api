# movie_similarity_flask_api
Welcome! This repo includes code to train a movie similarity recommender model, and host it with a flask app. The flask app creates an entpoint anyone can use to get movie recommendations! Here is how to set it up locally on your machine and get recommendations. 

## How to clone and run this service locally
1. Clone the repo to your local machine
2. Create a conda virtual enviroment for this repo. This is just so we're all running the same versions of things. :-) 
- `conda env create -f environment.yml`
- `source activate movie_similarity_flask_api`
3. Run the service
- `python app.py`
4. Make a curl request
`curl -X GET http://127.0.0.1:5000/ -d movie_title='Zodiac (2007)'`
You could also make requests to the serive using the `run_service.ipynb`. 
Sit back and enjoy one of your recommended movies!
![](https://media.giphy.com/media/eSA5lwLzcE2NW/giphy.gif)

# Model 
## Data
I used the tags and ratings data from the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/). 

## Model
I trained an item-item collaborative recommender system using the user-movie ratings, and an item-item content recommender system using the movie tags. I then ensembled both models by averaging the cosine the similarity scores from both the collaborative and the content latent embeddings. Collaborative recommenders have popularity bias, and will only be able to accuratly recommend a small percent of the movies, while missing the meaningfull recommendations that are less popular. In the context of movie recommendatiions, the collaborative method may not do a great job recommending movies you could'nt think of on your own. The content recommender will recommend on-topic movies, but these items may have a low overall rating. The content recommender will be able to recommend new titles you may have never heard of, but match the topic of the seed_movie. By ensembing the content and collaborative methods, I've created a model that recommends movies that have been rated similarly, while still matching the style of the seed_movie. 

Both models use SVD to calculate latent dimensions from the original matricies. These embeddings are used to calculate pairwise movie cosine similarity scores from each model, and the similarity scores are averaged to determine the final similarity score.

The Flask App returns the top 20 most similar movies for the movie entered. If you really enjoyed a movie and would love to watch more like it, simply submit it in the curl request to get some on-topic recommendations of movies others have also enjoyed!

### Collaborative Filter
The code for training the collaborative filter can be found in
`train_collaborative_similarity_recommenders.ipynb`
I reduced the ratings data to ony include users that have rated more than 10 movies. This reduced the sparsity for the SVD. Because I am only concered with similarity and I am not trying to predict user-movie ratings, I marked unrated movies as 0. I then compressed the movie-ratings matrix into 100 latent dimensions using SVD.

### Content Filter
The code for training the content filter can be found in
`train_content_similarity_model.ipynb`
I contactinated all the tag and genera data into one document per movie, and perfomed TF-IDF on these documents to extract tokenized features. I then compressed the features into 100 latent dimensions using SVD. 


