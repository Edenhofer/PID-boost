#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import os
import sys

import numpy as np
from keras.layers import Activation, Dense, Dropout, MaxPooling1D
from keras.models import Sequential
from keras.utils import to_categorical

import lib

try:
    import seaborn as sns

    # Enable and customize default plotting style
    sns.set_style("whitegrid")
except ImportError:
    pass

try:
    import argcomplete
except ImportError:
    pass


ParticleFrame = lib.ParticleFrame

# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_util = parser.add_argument_group('utility options', 'Parameters for altering the behavior of the program\'s input-output handling')
group_util.add_argument('-i', '--input', dest='input_directory', action='store', default='./',
                    help='Directory in which the program shall search for root files for each particle')
group_util.add_argument('-o', '--output', dest='output_directory', action='store', default='./res/',
                    help='Directory for the generated output (mainly plots); Skip saving plots if given \'/dev/null\'.')
group_util.add_argument('--interactive', dest='interactive', action='store_true', default=True,
                    help='Run interactively, i.e. show plots')
group_util.add_argument('--non-interactive', dest='interactive', action='store_false', default=True,
                    help='Run non-interactively and hence unattended, i.e. show no plots')
group_util.add_argument('-f', '--file', dest='output_file', action='store', default='./model.h5',
                    help='Path where the model should be saved to including the filename; Skip saving if given \'/dev/null\'.')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass


args = parser.parse_args()

# Evaluate the arguments
input_directory = args.input_directory
interactive = args.interactive
output_directory = args.output_directory
output_file = args.output_file

# Read in all the particle's information into a dictionary of pandas-frames
data = ParticleFrame(input_directory=input_directory, output_directory=output_directory, interactive=interactive)

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
model.add(Dense(len(labels) * 3, input_shape=(x.shape[1],), activation='relu', use_bias=True))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

# Compilation for a multi-class classification problem
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(y, num_classes=len(labels))
# Train the model
model.fit(x, one_hot_labels, epochs=epochs, batch_size=batch_size)

if output_file != '/dev/null':
    if not os.path.exists(os.path.dirname(output_file)):
        print('Creating desired parent directory "%s" for the output file "%s"'%(os.path.dirname(output_file), output_file), file=sys.stderr)
        os.makedirs(os.path.dirname(output_file), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation

    model.save(output_file)
