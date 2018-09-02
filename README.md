# movie_similarity_flask_api
Welcome! This repo includes code to train a movie similarity model, and host it with a flask app.
The code for training the recommender model is imported from our recommender system library. 

![](https://media.giphy.com/media/eSA5lwLzcE2NW/giphy.gif)

# How to train and run this service locally
1. clone the repo to your local machine
2. Create a conda virtual enviroment for this repo. This is just so we're all running the same versions of things. :-) 
- `conda env create -f environment.yml`
- `source activate movie_similarity_flask_api`
  
3. Train the model

4. Run the service
- `python app.py`
 
5. Make a curl request
`curl -X GET http://127.0.0.1:5000/ -d movie='test'`
You could also make requests to the serive using the `run_service.ipynb`. 
