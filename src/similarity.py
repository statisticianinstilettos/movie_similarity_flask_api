import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityModel(object):

    def get_similar_movies_from_embeddings(self, seed_movie, latent_features, movies):
        ''' Return similarity scores between the seed_movie and all other movies. '''

        #get style variant's feature vector in latent space
        item_vector = np.array(latent_features.loc[seed_movie]).reshape(1, -1)

        #calculate similarity
        similarities = cosine_similarity(latent_features, item_vector, dense_output=True)

        # get detailed movie info
        similarities = pd.DataFrame(similarities, index = latent_features.index.tolist())
        similarities.columns = ['similarity_score']
        sim_df = pd.merge(movies, similarities, left_index=True, right_index=True)
        sim_df.sort_values('similarity_score', ascending=False, inplace=True)

        return sim_df

    def ensemble_similarity_scores(self, collaborative_similarity, content_similarity):
        ''' Retrun top 20 movie recommendations from ensembled similarity scores'''
        #average both similarity scores
        sim_df_ensembled = pd.merge(collaborative_similarity, pd.DataFrame(content_similarity['similarity_score']), left_index=True, right_index=True)
        sim_df_ensembled['similarity_score'] = (sim_df_ensembled['similarity_score_x'] + sim_df_ensembled['similarity_score_y'])/2
        sim_df_ensembled.drop("similarity_score_x", axis=1, inplace=True)
        sim_df_ensembled.drop("similarity_score_y", axis=1, inplace=True)

        #sort by average similarity score
        sim_df_ensembled.sort_values('similarity_score', ascending=False, inplace=True)

        #remove recommendation of the seed movie
        sim_df_ensembled = sim_df_ensembled.iloc[1:]

        #round average ratings
        sim_df_ensembled['avg_rating'] = round(sim_df_ensembled['avg_rating'], 1)

        return sim_df_ensembled[['title', 'genres', 'avg_rating', 'similarity_score']].head(20).T.to_dict()
