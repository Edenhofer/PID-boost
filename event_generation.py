import basf2
from modularAnalysis import outputMdst
from reconstruction import add_reconstruction
from simulation import add_simulation

path = basf2.create_path()

# Add modules to the root basf2 path; Parameters can be retrieved using basf2.print_params(`moduleVar`)
setter = path.add_module("EventInfoSetter", evtNumList=[100000])
generator = path.add_module("EvtGenInput", userDECFile='b2_mc_event.dec')

add_simulation(path)
add_reconstruction(path)

outputMdst('b2_mc_events.mdst.root', path=path)

basf2.process(path)
