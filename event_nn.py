#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout, MaxPooling1D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import lib

try:
    import argcomplete
except ImportError:
    pass


ParticleFrame = lib.ParticleFrame

# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Train a configurable neural network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_action = parser.add_mutually_exclusive_group(required=True)
group_opt = parser.add_argument_group('sub-options', 'Parameters which make only sense to use in combination with an action and which possibly alters their behavior')
group_util = parser.add_argument_group('utility options', 'Parameters for altering the behavior of the program\'s input-output handling')
group_backend = parser.add_argument_group('backend options', 'Parameters for configuring the backend used for modelling and training')
group_action.add_argument('--pca', dest='run', action='store_const', default=None, const='pca',
                    help='Run the model on the principal components of the data')
group_action.add_argument('--pidProbability', dest='run', action='store_const', default=None, const='pidProbability',
                    help='Run the model on the `pidProbability` data by detector')
group_opt.add_argument('--batch-size', dest='batch_size', action='store', type=int, default=32,
                    help='Size of each batch')
group_opt.add_argument('--epoch', dest='epoch', action='store', type=int, default=10,
                    help='Number of iterations to train the model (epoch)')
group_opt.add_argument('--ncomponents', dest='n_components', action='store', type=int, default=12,
                    help='Number of components to keep after performing a PCA on the data')
group_opt.add_argument('--training-fraction', dest='training_fraction', action='store', type=float, default=0.8,
                    help='Fraction of the whole data which shall be used for training; Non-training data is used for validation')
group_util.add_argument('--module-file', dest='module_path', action='store', default=None,
                    help='Path including the filename where the model should be saved to; Skip saving if given \'/dev/null\'')
group_util.add_argument('--history-file', dest='history_path', action='store', default=None,
                    help='Path including the filename where the history of the model-fitting should be saved to; Skip saving if given \'/dev/null\'')
group_util.add_argument('-i', '--input', dest='input_directory', action='store', default='./',
                    help='Directory in which the program shall search for ROOT files for each particle')
group_util.add_argument('--input-pickle', dest='input_pickle', action='store', default=None,
                    help='Pickle file path containing a ParticleFrame object which shall be read in instead of ROOT files; Takes precedence when specified')
group_util.add_argument('-o', '--output', dest='output_directory', action='store', default='./res/',
                    help='Directory for the generated output (mainly plots); Skip saving plots if given \'/dev/null\'')
group_util.add_argument('--output-pickle', dest='output_pickle', action='store', default=None,
                    help='Pickle file path containing a ParticleFrame object which shall be read in instead of ROOT files; Takes precedence when specified')
group_util.add_argument('--interactive', dest='interactive', action='store_true', default=True,
                    help='Run interactively, i.e. show plots')
group_util.add_argument('--non-interactive', dest='interactive', action='store_false', default=True,
                    help='Run non-interactively and hence unattended, i.e. show no plots')
group_backend.add_argument('--cpu', dest='backend_cpu', nargs='?', action='store', type=int, default=1, const=1,
                    help='Whether to use the CPU for processing and optionally the device count; 0 disables its usage')
group_backend.add_argument('--gpu', dest='backend_gpu', nargs='?', action='store', type=int, default=0, const=1,
                    help='Whether to use the GPU for processing and optionally the device count; 0 disables its usage')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass


args = parser.parse_args()

# Evaluate the backend options
backend_gpu = args.backend_gpu
backend_cpu = args.backend_cpu
# Enforce the specified options using the tensorflow backend; Remember the GPU will always be initialized regardless
num_GPU = backend_gpu if backend_gpu else 0
num_CPU = backend_cpu if backend_cpu else 0
config = tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU': num_CPU, 'GPU': num_GPU})
session = tf.Session(config=config)
K.set_session(session)

# Evaluate the input-output arguments
input_directory = args.input_directory
input_pickle = args.input_pickle
interactive = args.interactive
output_directory = args.output_directory
output_pickle = args.output_pickle
history_path = args.history_path
module_path = args.module_path

# Read in all the particle's information into a dictionary of pandas-frames
if input_pickle:
    data = ParticleFrame(pickle_path=input_pickle, output_directory=output_directory, interactive=interactive)
else:
    data = ParticleFrame(input_directory=input_directory, output_directory=output_directory, interactive=interactive)

truth_color_column = 'mcPDG_color'
nn_color_column = 'nn_mcPDG'
detector = 'all' # Bad hardcoding stuff which should actually be configurable
# Evaluate sub-options
epochs = args.epoch
batch_size = args.batch_size
training_fraction = args.training_fraction
n_components = args.n_components

# Concatenate the whole data into one huge multi-indexable DataFrame
# Take special care about extracting the final result, since this is a copy
augmented_matrix = pd.concat(data.values(), keys=data.keys())

test_selection = np.random.choice([True, False], augmented_matrix.shape[0], p=[training_fraction, 1-training_fraction])
validation_selection = np.invert(test_selection) # Use everything not utilized for testing as validation data

# Assemble the array representing the desired output
# By not setting the labesl to list(np.unique(np.abs(augmented_matrix['mcPDG'].values))) we certainly missclassify some particles, however to be able to compare the epsilonPID matrices we should still have an absolute classification into the classes available to pidProbability; Include 'none' either way
labels = [0] + [lib.pdg_from_name_faulty(p) for p in ParticleFrame.particles]
for v in list(np.unique(np.abs(augmented_matrix['mcPDG'].values))):
    try:
        augmented_matrix.at[(augmented_matrix['mcPDG'] == v) | (augmented_matrix['mcPDG'] == -1 * v), truth_color_column] = labels.index(v)
    except ValueError:
        augmented_matrix.at[(augmented_matrix['mcPDG'] == v) | (augmented_matrix['mcPDG'] == -1 * v), truth_color_column] = labels.index(0)
target = augmented_matrix[truth_color_column]

# Assemble the input matrix on which to train the model
run = args.run
if run == 'pidProbability':
    design_columns = []
    for p in ParticleFrame.particles:
        for d in ParticleFrame.detectors + ParticleFrame.pseudo_detectors:
            design_columns += ['pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc']
    design_matrix = augmented_matrix[design_columns].fillna(0.) # Fill null in cells with no value (clean up probability columns)
elif run == 'pca':
    design_columns = list(set(augmented_matrix.keys()) - {'isSignal', 'mcPDG', 'mcErrors'})
    design_matrix = augmented_matrix[design_columns].fillna(0.) # Fill null in cells with no value (clean up probability columns)

    scaler = StandardScaler()
    scaler.fit(design_matrix[test_selection])
    scaler.transform(design_matrix)
    pca = PCA(n_components=n_components)
    pca.fit(design_matrix[test_selection])
    pca.transform(design_matrix)
    print('Selected principal components explain %.4f of the variance in the data'%(pca.explained_variance_ratio_.sum()))

x_test = design_matrix[test_selection].values
y_test = target[test_selection].values
x_validation = design_matrix[validation_selection].values
y_validation = target[validation_selection].values

# Layer selection
model = Sequential()
model.add(Dense(len(labels) * 2, input_shape=(x_test.shape[1],), activation='relu', use_bias=True))
model.add(Dropout(0.2))
model.add(Dense(len(labels) * 3, activation='relu', use_bias=True))
model.add(Dropout(0.2))
model.add(Dense(len(labels) * 2, activation='relu', use_bias=True))
model.add(Dense(int(len(labels) * 1.5), activation='relu', use_bias=True))
model.add(Dense(len(labels), activation='softmax'))

# Compilation for a multi-class classification problem
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
y_test_hot = to_categorical(y_test, num_classes=len(labels))
y_validation_hot = to_categorical(y_validation, num_classes=len(labels))

# Visualize the training
tensorboard_callback = TensorBoard(log_dir=os.path.join(output_directory, 'logs'), histogram_freq=1, batch_size=batch_size)
# Train the model
history = model.fit(x_test, y_test_hot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_hot), shuffle=True, callbacks=[tensorboard_callback])

score = model.evaluate(x_validation, y_validation_hot, batch_size=batch_size)
print('\nModel validation using independent data - loss: %.6f - acc: %.6f'%(score[0], score[1]))

# Add predictions of the validation set to the ParticleFrame instance
y_prediction_labels = np.argmax(model.predict(x_validation, batch_size=batch_size), axis=1)
y_prediction = [labels[i] for i in y_prediction_labels]
augmented_matrix[nn_color_column] = np.nan
augmented_matrix.at[validation_selection, nn_color_column] = y_prediction

# Add cutting columns baring the various particleIDs generated by the network to the object
cutting_columns = {k: 'nn_' + v for k, v in ParticleFrame.particleIDs.items()}
for p, c in cutting_columns.items():
    augmented_matrix.at[validation_selection, c] = 0.
    augmented_matrix.at[validation_selection & (augmented_matrix[nn_color_column] == lib.pdg_from_name_faulty(p)), c] = 1.

output_columns = [nn_color_column] + list(cutting_columns.values())
for p, particle_data in data.items():
    particle_data[output_columns] = augmented_matrix.loc[(p,)][output_columns]
    # Since sensible data has only been estimated for the validation set, it makes sense to also just save this portion and disregard the rest
    particle_data.dropna(subset=[nn_color_column], inplace=True)

# Set a sensibles default suffix for filenames
config = model.get_config()
savefile_suffix = run + '_nLayers' + str(len(config)) + '_nEpochs' + str(history.params['epochs']) + '.h5'
# Save the ParticleFrame data
if output_pickle != '/dev/null':
    if output_pickle == None:
        output_pickle = os.path.join(output_directory, data.__class__.__name__ + savefile_suffix + '.pkl')
    if not os.path.exists(os.path.dirname(output_pickle)):
        print('Creating desired parent directory "%s" for the particle-data pickle file "%s"'%(os.path.dirname(output_pickle), output_pickle), file=sys.stderr)
        os.makedirs(os.path.dirname(output_pickle), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    data.save(pickle_path=output_pickle)
# Save model-fitting history if requested
if history_path != '/dev/null':
    if history_path == None:
        history_path = os.path.join(output_directory, 'history_' + savefile_suffix + '.pkl')
    if not os.path.exists(os.path.dirname(history_path)):
        print('Creating desired parent directory "%s" for the model-fitting history pickle file "%s"'%(os.path.dirname(history_path), history_path), file=sys.stderr)
        os.makedirs(os.path.dirname(history_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    # Selectively specify which data to save as the whole object can not be pickled
    history_data = {'history': history.history, 'epoch': history.epoch, 'params': history.params, 'run': run, 'n_components': n_components, 'training_fraction': training_fraction}
    pickle.dump(history_data, open(history_path, 'wb'), pickle.HIGHEST_PROTOCOL)
# Save module if requested
if module_path != '/dev/null':
    if module_path == None:
        module_path = os.path.join(output_directory, 'model_' + savefile_suffix + '.h5')
    if not os.path.exists(os.path.dirname(module_path)):
        print('Creating desired parent directory "%s" for the output file "%s"'%(os.path.dirname(module_path), module_path), file=sys.stderr)
        os.makedirs(os.path.dirname(module_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    model.save(module_path)
