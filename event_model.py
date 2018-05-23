#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
import pickle

import matplotlib.pyplot as plt

import lib

try:
    import argcomplete
except ImportError:
    pass


ParticleFrame = lib.ParticleFrame

# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_util = parser.add_argument_group('utility options', 'Parameters for altering the behavior of the program\'s input-output handling')
group_util.add_argument('--history-file', dest='history_path', action='store', default=None, required=True,
                    help='Path including the filename where the history of the model-fitting should be saved to; Skip saving if given \'/dev/null\'')
group_util.add_argument('-o', '--output', dest='output_directory', action='store', default='./res/',
                    help='Directory for the generated output (mainly plots); Skip saving plots if given \'/dev/null\'')
group_util.add_argument('--interactive', dest='interactive', action='store_true', default=True,
                    help='Run interactively, i.e. show plots')
group_util.add_argument('--non-interactive', dest='interactive', action='store_false', default=True,
                    help='Run non-interactively and hence unattended, i.e. show no plots')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass


args = parser.parse_args()

# Evaluate the input-output arguments
interactive = args.interactive
output_directory = args.output_directory
history_path = args.history_path

# Initialize an empty ParticleFrame and load the pickle_file
data = ParticleFrame(output_directory=output_directory, interactive=interactive)
history = pickle.load(open(history_path, 'rb'))

for key, title, ylabel in [('val_loss', 'Validation Loss', 'Loss'), ('val_acc', 'Validation Accuracy', 'Accuracy'), ('loss', 'Training Loss', 'Loss'), ('acc', 'Training Accuracy', 'Accuracy')]:
    plt.figure()
    plt.plot(history['epoch'], history['history'][key], color='orange')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    data.pyplot_sanitize_show(title, savefig_prefix='Neural Network Model %s: '%(history['run']), savefig_suffix=','.join([' with %d %s'%(value, key) for key, value in [('epochs', history['params']['epochs']), ('batch_size', history['params']['batch_size'])]]))

plt.show()
