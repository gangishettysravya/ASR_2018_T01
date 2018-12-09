import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
import leven

phones = ['space','aa','ae','ah','aw','ay','b','ch','d','dh','dx','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','sil','t','th','uh','uw','v','w','y','z']


def get_accuracy(features,labels,models):
	pred_prob = []
	for i in range(len(phones)):
		y = models[phones[i]].score_samples(features)
		pred_prob.append(y)

	pred_prob = np.array(pred_prob)
	predictions = np.argmax(pred_prob,axis=0)
	preds = [] #predictions labels
	
	count = 0
	for i in range(len(predictions)):
		preds.append(phones[predictions[i]])
		if phones[predictions[i]] == labels[i]:
			count = count+1

	acc = (count/features.shape[0])*100

	return acc,preds


#test_data

df = pd.read_hdf("./features/mfcc/test.hdf")
#print(df.head())
mfcc_features = np.array(df["features"].tolist())
mfcc_labels = np.array(df["labels"].tolist())

df = pd.read_hdf("./features/mfcc_delta/test.hdf")
mfcc_delta_features = np.array(df["features"].tolist())
mfcc_delta_labels = np.array(df["labels"].tolist())

df = pd.read_hdf("./features/mfcc_delta_delta/test.hdf")
mfcc_delta_delta_features = np.array(df["features"].tolist())
mfcc_delta_delta_labels = np.array(df["labels"].tolist())

with open('results/predictions_ground_truths/ground_truths.txt','w+') as f:
	for item in mfcc_labels:
		f.write("%s\n"% item)

#with energy coefficients mfcc 64 gmm
models = {}
for phone in phones:
	models[phone] = joblib.load("models/mfcc/with_energy_coefficients/64/"+phone+".pkl")
accu,preds = get_accuracy(mfcc_features,mfcc_labels,models)
print("Accuracy for mfcc with_energy_coefficients 64 mixture model :"+ str(accu))
with open('results/predictions_ground_truths/preds_64_mfcc_energy.txt','w+') as f:
	for item in preds:
		f.write("%s\n"% item)

#with energy coefficients delta_mfcc 64 gmm
models = {}
for phone in phones:
	models[phone] = joblib.load("models/mfcc_delta/with_energy_coefficients/64/"+phone+".pkl")
accu,preds = get_accuracy(mfcc_delta_features,mfcc_delta_labels,models)
print("Accuracy for mfcc_delta with_energy_coefficients 64 mixture model :"+ str(accu))
with open('results/predictions_ground_truths/preds_64_mfcc_delta_energy.txt','w+') as f:
	for item in preds:
		f.write("%s\n"% item)


#with energy coefficients delta_delta_mfcc 64 gmm
models = {}
for phone in phones:
	models[phone] = joblib.load("models/mfcc_delta_delta/with_energy_coefficients/64/"+phone+".pkl")
accu,preds = get_accuracy(mfcc_delta_delta_features,mfcc_delta_delta_labels,models)
print("Accuracy for mfcc_delta_delta with_energy_coefficients 64 mixture model :"+ str(accu))
with open('results/predictions_ground_truths/preds_64_mfcc_delta_delta_energy.txt','w+') as f:
	for item in preds:
		f.write("%s\n"% item)


#without energy coefficients delta_mfcc 64 gmm
models = {}
for phone in phones:
	models[phone] = joblib.load("models/mfcc_delta/without_energy_coefficients/64/"+phone+".pkl")
accu,preds = get_accuracy(mfcc_delta_features,mfcc_delta_labels,models)
print("Accuracy for mfcc_delta without_energy_coefficients 64 mixture model :"+ str(accu))
with open('results/predictions_ground_truths/preds_64_mfcc_delta_withoutenergy.txt','w+') as f:
	for item in preds:
		f.write("%s\n"% item)


#without energy coefficients delta_delta_mfcc 64 gmm
models = {}
for phone in phones:
	models[phone] = joblib.load("models/mfcc_delta_delta/without_energy_coefficients/64/"+phone+".pkl")
accu,preds = get_accuracy(mfcc_delta_delta_features,mfcc_delta_delta_labels,models)
print("Accuracy for mfcc_delta_delta without_energy_coefficients 64 mixture model :"+ str(accu))
with open('results/predictions_ground_truths/preds_64_mfcc_delta_delta_withoutenergy.txt','w+') as f:
	for item in preds:
		f.write("%s\n"% item)

mixtures = 2
while mixtures <= 256:
	models = {}
	for phone in phones:
		models[phone] = joblib.load("models/mfcc/without_energy_coefficients/"+str(mixtures)+"/"+phone+".pkl")
	accu,preds = get_accuracy(mfcc_features,mfcc_labels,models)
	print("Accuracy for mfcc without_energy_coefficients" + str(mixtures) +" mixture model :"+ str(accu))
	with open('results/predictions_ground_truths/preds_'+str(mixtures)+'_mfcc_withoutenergy.txt','w+') as f:
		for item in preds:
			f.write("%s\n"% item)
	mixtures *= 2
