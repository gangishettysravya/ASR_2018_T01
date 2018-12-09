import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture


path = "mfcc"
df = pd.read_hdf("./features/"+path+"/timit.hdf")
print(df.head())
mfcc_features = np.array(df["features"].tolist())
mfcc_labels = np.array(df["labels"].tolist())
#print(mfcc_features.shape)

phones = list(set(mfcc_labels))
phones.remove('')
phones.append('space')

#Reading the features from the phone-wise .hdf files

mfcc_dictionary = {}
mfcc_delta_dictionary = {}
mfcc_delta_delta_dictionary = {}

for phone in phones:
	mfcc_ph_df = pd.read_hdf("./features/mfcc/phone_wise/"+phone+".hdf")
	delta_ph_df = pd.read_hdf("./features/mfcc_delta/phone_wise/"+phone+".hdf")
	delta_delta_ph_df = pd.read_hdf("./features/mfcc_delta_delta/phone_wise/"+phone+".hdf")
	mfcc_dictionary[phone] = np.array(mfcc_ph_df["features"].tolist())[0]
	mfcc_delta_dictionary[phone] = np.array(delta_ph_df["features"].tolist())[0]
	mfcc_delta_delta_dictionary[phone] = np.array(delta_delta_ph_df["features"].tolist())[0]

print(mfcc_dictionary['space'].shape)
print(mfcc_delta_dictionary["space"].shape)
print(mfcc_delta_delta_dictionary["space"].shape)

gmm = GaussianMixture(n_components=64,covariance_type="diag",tol=0.001)


#Training and dumping for MFCC with energy coefficients for 64 mixture model a(i)
for phone in phones:
	phone_features = mfcc_dictionary[phone]
	gmm_model = gmm.fit(X=phone_features)
	joblib.dump((gmm_model),"models/mfcc/with_energy_coefficients/64/"+phone+".pkl",compress=3)
#	print(phone)
#	print(phone_features.shape)


#Training and dumping for MFCC+delta with energy coefficients for 64 mixture model b(i)
for phone in phones:
	phone_features = mfcc_delta_dictionary[phone]
	gmm_model = gmm.fit(X=phone_features)
	joblib.dump((gmm_model),"models/mfcc_delta/with_energy_coefficients/64/"+phone+".pkl",compress=3)
#	print(phone)
#	print(phone_features.shape)


#Training and dumping for MFCC_delta_delta with energy coefficients for 64 mixture model b(i)
for phone in phones:
	phone_features = mfcc_delta_delta_dictionary[phone]
	gmm_model = gmm.fit(X=phone_features)
	joblib.dump((gmm_model),"models/mfcc_delta_delta/with_energy_coefficients/64/"+phone+".pkl",compress=3)
#	print(phone)
#	print(phone_features.shape)

#Training and dumping for MFCC_delta without energy coefficient for 64 mixture b(ii)
for phone in phones:
	phone_features = mfcc_delta_dictionary[phone]
	phone_features[:,0] = 0
	phone_features[:,13] = 0
	gmm_model = gmm.fit(X=phone_features)
	joblib.dump((gmm_model),"models/mfcc_delta/without_energy_coefficients/64/"+phone+".pkl",compress=3)


#Training and dumping for MFCC_delta_delta without energy coefficient for 64 mixture model(ii)
for phone in phones:
	phone_features = mfcc_delta_delta_dictionary[phone]
	phone_features[:,0] = 0
	phone_features[:,13] = 0
	phone_features[:,26] = 0
	gmm_model = gmm.fit(X=phone_features)
	joblib.dump((gmm_model),"models/mfcc_delta_delta/without_energy_coefficients/64/"+phone+".pkl",compress=3)

#Training and dumping for MFCC without energy coefficient a(ii)

num_mixtures = 2
while num_mixtures<=256:
	gmm = GaussianMixture(n_components=num_mixtures,covariance_type="diag",tol=0.001)
	for phone in phones:
		phone_features = mfcc_dictionary[phone]
		phone_features[:,0] = 0
		gmm_model = gmm.fit(X=phone_features)
		joblib.dump((gmm_model),"models/mfcc/without_energy_coefficients/"+str(num_mixtures)+"/"+phone+".pkl",compress=3)
	num_mixtures *= 2