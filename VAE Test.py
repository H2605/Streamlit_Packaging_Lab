import streamlit as st
import pandas as pd
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt


latent_dim = 4
# Laden des gespeicherten Modells
autoencoder = load_model('/Users/huyduc/Documents/GitHub/Packaging Lab/mein_autoencoder2.h5')

#num_features = 3486
#input_shape = (num_features,)
#inputs = Input(shape=input_shape)
#x = Dense(64, activation='relu')(inputs)
#z_mean = Dense(latent_dim)(x)
#z_log_var = Dense(latent_dim)(x)
#def sampling(args):
#    z_mean, z_log_var = args
#    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
#                              mean=0., stddev=1.)
#    return z_mean + K.exp(z_log_var / 2) * epsilon

# Use the sampling function to create a layer that samples from the latent space
#z = Lambda(sampling)([z_mean, z_log_var])
# Define the decoder network
decoder = load_model('/Users/huyduc/Documents/GitHub/Packaging Lab/mein_decoder2.h5')
#decoder_inputs = Input(shape=(latent_dim,))
#x = Dense(64, activation='relu')(decoder_inputs)
#outputs = Dense(num_features, activation='sigmoid')(x)

# Define the autoencoder model
#encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#decoder = Model(decoder_inputs, outputs, name='decoder')
#outputs = decoder(encoder(inputs)[2])

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
with open('/Users/huyduc/Documents/GitHub/Packaging Lab/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


number = st.number_input('Wie Dateneinträge sollen generiert werden?', min_value=1, max_value=100000, value=1, step=1, format=None, key=None)
#st.write('The current number is ', number)
n_samples = number
latent_samples = np.random.normal(size=(n_samples, latent_dim))
arr_abs = np.absolute(latent_samples)
generated_data = decoder.predict(latent_samples)

spalten = pd.read_csv("/Users/huyduc/Documents/GitHub/Packaging Lab/kadoh32_1.csv")


generierte_daten = pd.DataFrame(generated_data, columns=spalten.columns)


scaler_df=scaler.inverse_transform(generierte_daten)
#generierte_daten[num_cols]=scaler_df
generierte_daten = pd.DataFrame(scaler_df, columns=spalten.columns)

num_cols=generierte_daten.iloc[:, -4:-1].columns
cat_cols=generierte_daten.iloc[:, :-4].columns
generierte_daten[num_cols]=np.where(generierte_daten[num_cols]<0.01,0,generierte_daten[num_cols])
generierte_daten[cat_cols]=np.where(generierte_daten[cat_cols]<1,0,1)
generierte_daten["Transportschaden"]=np.where(generierte_daten["Transportschaden"]<0.01,0,generierte_daten["Transportschaden"])
generierte_daten
#generierte_daten["Transportschaden"].hist()
fig, ax = plt.subplots()
ax.hist(generierte_daten["Transportschaden"], bins=20)

st.write('Verteilung der Transportschäden')
st.pyplot(fig)