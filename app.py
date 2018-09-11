import pickle
import pandas as pd
from src.similarity import SimilarityModel
from flask import Flask
from flask_restful import reqparse, Api, Resource
app = Flask(__name__)
api = Api(app)


#load enbeddings and movie details
text_embeddings = pd.read_pickle('data/svd_text_embeddings.pkl')
ratings_embeddings = pd.read_pickle('data/svd_ratings_embeddings.pkl')
movies = pd.read_csv('data/movie_demographics.csv', index_col=0)

#initialize similarity model
sm = SimilarityModel()

#parse payload
parser = reqparse.RequestParser()
parser.add_argument('movie_title')

class MovieSimilarity(Resource):
    def get(self):

        #get movie id from title
        #TODO update this to a string matching algorithm that finds the id from the string you typed.
        seed_title = parser.parse_args().movie_title
        seed_id = movies.query('title == @seed_title').index.values[0]

        #get similarity from the collaborative recommender
        collaborative_similarity = sm.get_similar_movies_from_embeddings(seed_id, ratings_embeddings, movies)

        #get similarity from the collaborative recommender
        content_similarity = sm.get_similar_movies_from_embeddings(seed_id, text_embeddings, movies)

        #ensemble similarity scores and return top 20 most similar movies
        output = sm.ensemble_similarity_scores(collaborative_similarity, content_similarity)
        return output

api.add_resource(MovieSimilarity, '/')

if __name__ == '__main__':
    app.run(debug=False)
