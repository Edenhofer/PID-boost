import basf2
import modularAnalysis
import pdg

path = basf2.create_path()

modularAnalysis.inputMdstList("default", ['sample.mdst.root'], path=path)
modularAnalysis.fillParticleLists([('K+', ''), ('pi+', ''), ('mu+', '')], path=path)
modularAnalysis.reconstructDecay('D0 -> K- pi+', '', path=path)
modularAnalysis.fitVertex('D0', 0., path=path)
modularAnalysis.matchMCTruth('D0', path=path)

# Specify the particle of interest by Particle Data Group Code (pdgCode)
pdg_code = pdg.from_name('K-')
# Specify the detector of interest by name, see the 'parseDetectors' function in the C++ code
detector = 'top'

# Export variables of the analysis to NTuple root file
# Inspect the value using modularAnalysis.printVariableValues('K+', `varName(varArg)`, path=path)
modularAnalysis.variablesToNTuple('D0', ['Mbc', 'deltaE', 'M', 'p', 'E', 'useCMSFrame(p)', 'useCMSFrame(E)', 'daughter(0, kaonID)', 'daughter(1, pionID)', 'isSignal', 'mcErrors'], filename='sample.root', path=path)
modularAnalysis.variablesToNTuple('K+', 'pidLogLikelihoodValueExpert(' + str(pdg_code) + ', ' + detector + ')', filename='K+_sample.root', path=path)

basf2.process(path)
