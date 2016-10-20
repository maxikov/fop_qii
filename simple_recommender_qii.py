# Simple QII on recommender system test
# usage: python simple_recommender_qii.py <user-name>
# where <user-name> can be 'Toby', etc.

import sys

from scikits.crab import datasets
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

import numpy
import random

random.seed()

# Set up a recommendation system
movies = datasets.load_sample_movies()
model = MatrixPreferenceDataModel(movies.data)
similarity = UserSimilarity(model, pearson_correlation)
recommender = UserBasedRecommender(model, similarity, with_preference=True)

print movies.data
print movies.user_ids

average_local_inf = {}
iters = 5  # More iterations, greater accuracy

user_index = -1
for id in movies.user_ids:
	if movies.user_ids[id] == sys.argv[1]:
		user_index = id

if user_index < 0:
	print 'User not found'
	sys.exit()

prediction = recommender.recommend(user_index)
print "Prediction: ", prediction

user_pref = model.preferences_from_user(user_index)
print "User preferences: ", user_pref
for item in user_pref:
	local_influence = [0.0]*len(prediction)
	item_index = item[0]
	for i in xrange(iters):
		new_pref = random.random()*4.0 + 1.0
		model.set_preference(user_index, item_index, new_pref)
		new_prediction = [recommender.estimate_preference(user_index, id[0]) for id in prediction]
		print 'new prediction: ', new_prediction
		for p in xrange(len(prediction)):
			if not numpy.isnan(new_prediction[p]):
				local_influence[p] = local_influence[p] + abs(new_prediction[p]-prediction[p][1])

	print local_influence
	average_local_inf[item] = sum(local_influence)/(iters*len(prediction))
	print('Average local influence %s: %.3f' % (item, average_local_inf[item]))






