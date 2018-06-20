#!/usr/bin/env python3

# For consistency with other MC9 data it is important to use the same release: In order to do so invoke `setuprel release-01-00-03`
# Furthermore it might make sense to first compile the data once with the environment variable and once without it (`BELLE2_CONDB_GLOBALTAG="Calibration_Offline_Development"`)

from basf2 import *
from generators import add_evtgen_generator
from simulation import add_simulation
from reconstruction import add_reconstruction, add_mdst_output
from L1trigger import add_tsim
from ROOT import Belle2
import glob

# Background (collision) files
bg = glob.glob('/srv/data/*/bkg/*.root')

# Number of events to generate, can be overridden with `-n`
num_events = 10000
# Output filename, can be overridden with `-o`
output_filename = "RootOutput.root"

# Create path
main = create_path()

# Specify number of events to be generated
main.add_module("EventInfoSetter", expList=0, runList=0, evtNumList=num_events)

# Generate BBbar events
add_evtgen_generator(main, finalstate='charged')

# Detector simulation
add_simulation(main, bkgfiles=bg)

# Trigger simulation
add_tsim(main)

# Reconstruction
add_reconstruction(main)

# Finally add mdst output
add_mdst_output(main, filename=output_filename, additionalBranches=['TRGGDLResults', 'KlIds', 'KLMClustersToKlIds'])

# Process events and print call statistics
process(main)
print(statistics)
