import pickle
import pandas as pd
from src.similarity import SimilarityModel
from flask import Flask
from flask_restful import reqparse, Api, Resource
app = Flask(__name__)
api = Api(app)


#load enbeddings and movie details
svd_text_embeddings = pd.read_csv('data/svd_text_embeddings.csv', index_col=0)
svd_ratings_embeddings = pd.read_csv('data/svd_ratings_embeddings.csv', index_col=0)
nn_text_embeddings = pd.read_csv('data/nn_text_embeddings.csv', index_col=0)
nn_ratings_embeddings = pd.read_csv('data/nn_ratings_embeddings.csv', index_col=0)
movies = pd.read_csv('data/movie_demographics.csv', index_col=0)

#initialize similarity model
sm = SimilarityModel()

#parse payload
parser = reqparse.RequestParser()
parser.add_argument('movie_title')
parser.add_argument('model')

class MovieSimilarity(Resource):
    def get(self):
        #get movie id from title
        #TODO update this to a string matching algorithm that finds the id from the string you typed.
        seed_title = parser.parse_args().movie_title
        seed_id = movies.query('title == @seed_title').index.values[0]

        #get model type
        model = parser.parse_args().model
        assert model in ['matrix factorization', 'neural network'], "model must be svd or neural_network"

        if model == "matrix factorization":
            #get similarity from the collaborative recommender
            collaborative_similarity = sm.get_similar_movies_from_embeddings(seed_id, svd_ratings_embeddings, movies)
            #get similarity from the collaborative recommender
            content_similarity = sm.get_similar_movies_from_embeddings(seed_id, svd_text_embeddings, movies)
        if model == "neural network":
            #get similarity from the collaborative recommender
            collaborative_similarity = sm.get_similar_movies_from_embeddings(seed_id, nn_ratings_embeddings, movies)
            #get similarity from the collaborative recommender
            content_similarity = sm.get_similar_movies_from_embeddings(seed_id, nn_text_embeddings, movies)

        #ensemble similarity scores and return top 20 most similar movies
        output = sm.ensemble_similarity_scores(collaborative_similarity, content_similarity)

        return output

api.add_resource(MovieSimilarity, '/')

if __name__ == '__main__':
    app.run(debug=False)
