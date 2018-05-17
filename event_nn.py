#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import numpy as np
from keras.layers import Activation, Dense, Dropout, MaxPooling1D
from keras.models import Sequential
from keras.utils import to_categorical

import lib


ParticleFrame = lib.ParticleFrame
data = ParticleFrame(input_directory='./', output_directory='./res')

truth_color_column = 'mcPDG_color'

# Bad hardcoding stuff
detector = 'all'
p = 'K+'
particle_data = data[p]
batch_size = 32
epochs = 10

design_columns = []
for p in ParticleFrame.particles:
    for d in ParticleFrame.detectors + ParticleFrame.pseudo_detectors:
        design_columns += ['pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc']

labels = list(np.unique(np.abs(particle_data['mcPDG'].values)))
for v in labels:
    particle_data.at[(particle_data['mcPDG'] == v) | (particle_data['mcPDG'] == -1 * v), truth_color_column] = labels.index(v)

augmented_matrix = particle_data[design_columns + [truth_color_column]]
augmented_matrix = augmented_matrix.dropna(subset=[truth_color_column]) # Drop rows with no mcPDG code
augmented_matrix = augmented_matrix.fillna(0.) # Fill null values in probability columns (design matrix)

x = augmented_matrix[design_columns].values
y = augmented_matrix[truth_color_column].values

# Layer selection
model = Sequential()
model.add(Dense(x.shape[1], input_shape=(x.shape[1],), activation='relu', use_bias=True))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

# Compilation for a multi-class classification problem
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(y, num_classes=len(labels))
# Train the model
model.fit(x, one_hot_labels, epochs=epochs, batch_size=batch_size)
