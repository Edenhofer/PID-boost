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
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Activation, Dense, Dropout, MaxPooling1D
from keras.models import Sequential, load_model
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
group_opt.add_argument('--epochs', dest='epochs', action='store', type=int, default=10,
                    help='Number of iterations to train the model (epochs)')
group_opt.add_argument('--ncomponents', dest='n_components', action='store', type=int, default=12,
                    help='Number of components to keep after performing a PCA on the data')
group_opt.add_argument('--sampling-method', dest='sampling_method', action='store', choices=['fair', 'biased'], default='fair',
                    help='Specification of which sampling method should be applied')
group_opt.add_argument('--training-fraction', dest='training_fraction', action='store', type=float, default=0.8,
                    help='Fraction of the whole data which shall be used for training; Non-training data is used for validation')
group_util.add_argument('--module-file', dest='module_path', action='store', default=None,
                    help='Path including the filename where the model should be saved to; Skip saving if given \'/dev/null\'')
group_util.add_argument('--checkpoint-file', dest='checkpoint_path', action='store', default=None,
                    help='Path including the filename where to periodically save the best performing module to; Skip saving if given \'/dev/null\'')
group_util.add_argument('--history-file', dest='history_path', action='store', default=None,
                    help='Path including the filename where the history of the model-fitting should be saved to; Skip saving if given \'/dev/null\'')
group_util.add_argument('-i', '--input', dest='input_directory', action='store', default='./',
                    help='Directory in which the program shall search for ROOT files for each particle')
group_util.add_argument('--input-pickle', dest='input_pickle', action='store', default=None,
                    help='Pickle file path containing a ParticleFrame object which shall be read in instead of ROOT files; Takes precedence when specified')
group_util.add_argument('--input-module', dest='input_module', action='store', default=None,
                    help='Path to a module file to use instead of a self-constructed one')
group_util.add_argument('--log-dir', dest='log_directory', action='store', default=None,
                    help='Directory where TensorFlow is supposed to log to; Skip saving if given \'/dev/null\'')
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
input_module = args.input_module
interactive = args.interactive
log_directory = args.log_directory
output_directory = args.output_directory
output_pickle = args.output_pickle
history_path = args.history_path
module_path = args.module_path
checkpoint_path = args.checkpoint_path

# Read in all the particle's information into a dictionary of pandas-frames
if input_pickle:
    data = ParticleFrame(pickle_path=input_pickle, output_directory=output_directory, interactive=interactive)
else:
    data = ParticleFrame(input_directory=input_directory, output_directory=output_directory, interactive=interactive)

spoiling_columns = {'genMotherID', 'genMotherP', 'genMotherPDG', 'genParticleID', 'isExtendedSignal', 'isPrimarySignal', 'isSignal', 'isSignalAcceptMissingNeutrino', 'mcDX', 'mcDY', 'mcDZ', 'mcDecayTime', 'mcE', 'mcErrors', 'mcFSR', 'mcISR', 'mcInitial', 'mcMatchWeight', 'mcP', 'mcPDG', 'mcPX', 'mcPY', 'mcPZ', 'mcPhotos', 'mcRecoilMass', 'mcVirtual', 'nMCMatches'}
truth_color_column = 'mcPDG_color'
sampling_weight_column = 'sampling_weight'
nn_color_column = 'nn_mcPDG'
detector = 'all' # Bad hardcoding stuff which should actually be configurable
# Evaluate sub-options
epochs = args.epochs
batch_size = args.batch_size
training_fraction = args.training_fraction
n_components = args.n_components
sampling_method = args.sampling_method

# Concatenate the whole data into one huge multi-indexable DataFrame
# Take special care about extracting the final result, since this is a copy
augmented_matrix = pd.concat(data.values(), keys=data.keys())

# Assemble the array representing the desired output
# By not setting the labels to list(np.unique(np.abs(augmented_matrix['mcPDG'].values))) we certainly missclassify some particles, however to be able to compare the epsilonPID matrices we should still have an absolute classification into the classes available to pidProbability; Include 'none' either way
labels = [0] + [lib.pdg_from_name_faulty(p) for p in ParticleFrame.particles]
for v in list(np.unique(np.abs(augmented_matrix['mcPDG'].values))):
    try:
        augmented_matrix.at[(augmented_matrix['mcPDG'] == v) | (augmented_matrix['mcPDG'] == -1 * v), truth_color_column] = labels.index(v)
    except ValueError:
        augmented_matrix.at[(augmented_matrix['mcPDG'] == v) | (augmented_matrix['mcPDG'] == -1 * v), truth_color_column] = labels.index(0)
augmented_matrix[truth_color_column] = augmented_matrix[truth_color_column].astype(int)

if sampling_method == 'fair':
    augmented_matrix[sampling_weight_column] = 1. / augmented_matrix.groupby(truth_color_column)[truth_color_column].transform('count')
elif sampling_method == 'biased':
    augmented_matrix[sampling_weight_column] = 1.
# Allow sampling rows multiple times to not limit the sample size too much by the particles with a lower abundance
test_selection = augmented_matrix.sample(frac=training_fraction, weights=sampling_weight_column, replace=True).index
# Use everything not utilized for testing as validation data
validation_selection = augmented_matrix.drop(test_selection).index
n_duplicated = test_selection.duplicated()
print('Sampled test data contains %d duplicated rows (%.4f%%) (e.g. due to fair particle treatment)'%(sum(n_duplicated), sum(n_duplicated)/test_selection.shape[0]*100))

# Assemble the input matrix on which to train the model
run = args.run
if run == 'pidProbability':
    design_columns = []
    for p in ParticleFrame.particles:
        for d in ParticleFrame.detectors + ParticleFrame.pseudo_detectors:
            design_columns += ['pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc']
    design_matrix = augmented_matrix[design_columns].fillna(0.) # Fill null in cells with no value (clean up probability columns)
elif run == 'pca':
    design_columns = list(set(augmented_matrix.keys()) - spoiling_columns)
    design_matrix = augmented_matrix[design_columns].fillna(0.) # Fill null in cells with no value (clean up probability columns)

    scaler = StandardScaler()
    scaler.fit(design_matrix.loc[test_selection])
    scaler.transform(design_matrix)
    pca = PCA(n_components=n_components)
    pca.fit(design_matrix.loc[test_selection])
    pca.transform(design_matrix)
    print('Selected principal components explain %.4f of the variance in the data'%(pca.explained_variance_ratio_.sum()))
x_test = design_matrix.loc[test_selection].values
y_test = augmented_matrix.loc[test_selection][truth_color_column].values
x_validation = design_matrix.loc[validation_selection].values
y_validation = augmented_matrix.loc[validation_selection][truth_color_column].values

# Convert labels to categorical one-hot encoding
y_test_hot = to_categorical(y_test, num_classes=len(labels))
y_validation_hot = to_categorical(y_validation, num_classes=len(labels))

# Layer selection
if input_module:
    print('Loading input module from "%s"'%(input_module))
    model = load_model(input_module)
else:
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

# Set a sensibles default suffix for filenames
config = model.get_config()
savefile_suffix = run + '_' + sampling_method + '_nLayers' + str(len(config)) + '_nEpochs' + str(epochs)

# Visualize and save the training
keras_callbacks = []
if log_directory != '/dev/null':
    if log_directory == None:
        log_directory = os.path.join(output_directory, 'logs')
    if not os.path.exists(log_directory):
        print('Creating desired log directory "%s"'%(log_directory), file=sys.stderr)
        os.makedirs(log_directory, exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    keras_callbacks += [TensorBoard(log_dir=log_directory, histogram_freq=1, batch_size=batch_size)]
if checkpoint_path != '/dev/null':
    if checkpoint_path == None:
        checkpoint_path = os.path.join(output_directory, 'model_checkpoint_' + savefile_suffix + '.h5')
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        print('Creating desired parent directory "%s" for the checkpoint file "%s"'%(os.path.dirname(checkpoint_path), checkpoint_path), file=sys.stderr)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    keras_callbacks += [ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)]

# Train the model
history = model.fit(x_test, y_test_hot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_hot), shuffle=True, callbacks=keras_callbacks)

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
    augmented_matrix.at[augmented_matrix[nn_color_column] == abs(lib.pdg_from_name_faulty(p)), c] = 1.

output_columns = [nn_color_column, sampling_weight_column] + list(cutting_columns.values())
for p, particle_data in data.items():
    particle_data[output_columns] = augmented_matrix.loc[(p,)][output_columns]
    # Since sensible data has only been estimated for the validation set, it makes sense to also just save this portion and disregard the rest
    particle_data.dropna(subset=[nn_color_column], inplace=True)

# Save the ParticleFrame data
if output_pickle != '/dev/null':
    if output_pickle == None:
        output_pickle = os.path.join(output_directory, data.__class__.__name__ + '_' + savefile_suffix + '.pkl')
    if not os.path.exists(os.path.dirname(output_pickle)):
        print('Creating desired parent directory "%s" for the particle-data pickle file "%s"'%(os.path.dirname(output_pickle), output_pickle), file=sys.stderr)
        os.makedirs(os.path.dirname(output_pickle), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    data.descriptions['nn'] = savefile_suffix.replace('_', ' ')
    data.save(pickle_path=output_pickle)
# Save model-fitting history if requested
if history_path != '/dev/null':
    if history_path == None:
        history_path = os.path.join(output_directory, 'history_' + savefile_suffix + '.pkl')
    if not os.path.exists(os.path.dirname(history_path)):
        print('Creating desired parent directory "%s" for the model-fitting history pickle file "%s"'%(os.path.dirname(history_path), history_path), file=sys.stderr)
        os.makedirs(os.path.dirname(history_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    # Selectively specify which data to save as the whole object can not be pickled
    history_data = {'history': history.history, 'epoch': history.epoch, 'params': history.params, 'run': run, 'n_components': n_components, 'n_Layers': str(len(config)), 'training_fraction': training_fraction, 'sampling_method': sampling_method, 'savefile_suffix': savefile_suffix.replace('_', ' ')}
    pickle.dump(history_data, open(history_path, 'wb'), pickle.HIGHEST_PROTOCOL)
# Save module if requested
if module_path != '/dev/null':
    if module_path == None:
        module_path = os.path.join(output_directory, 'model_' + run + '_nLayers' + str(len(config)) + '.h5')
    if not os.path.exists(os.path.dirname(module_path)):
        print('Creating desired parent directory "%s" for the output file "%s"'%(os.path.dirname(module_path), module_path), file=sys.stderr)
        os.makedirs(os.path.dirname(module_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation
    model.save(module_path)
