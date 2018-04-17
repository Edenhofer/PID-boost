import basf2
import modularAnalysis
import pdg
import variables

path = basf2.create_path()
rootVars = []

# Specify the decay of interest
vertex_parent = 'D0'
particles_of_interest = ['K-', 'pi+']
rootVars += ['daughter(0, kaonID)', 'daughter(1, pionID)']
decay = vertex_parent + ' -> ' + ' '.join(particles_of_interest)

detector_list = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

var_list = []
for i, p in enumerate(particles_of_interest):
    for d in detector_list:
        var_list += ['pidLogLikelihoodValueExpert_' + 'daughter' + str(i) + '_' + d + 'Detector']
        variables.variables.addAlias(var_list[-1], 'daughter(' + str(i) + ', ' + 'pidLogLikelihoodValueExpert(' + str(pdg.from_name('K-')) + ', ' + str(d) + ')' + ')')

variables.variables.addCollection('pidLogLikelihoodValueExpertDaughtersByDetector', variables.std_vector(*var_list)) 
rootVars += ['pidLogLikelihoodValueExpertDaughtersByDetector']

modularAnalysis.inputMdstList("default", ['sample.mdst.root'], path=path)
modularAnalysis.fillParticleLists([(p, '') for p in particles_of_interest], path=path)
modularAnalysis.reconstructDecay(decay, '', path=path)
modularAnalysis.fitVertex(vertex_parent, 0., path=path)
modularAnalysis.matchMCTruth(vertex_parent, path=path)

# Export variables of the analysis to NTuple root file
# Inspect the value using modularAnalysis.printVariableValues('K+', `varName(varArg)`, path=path)
rootVars += ['isSignal', 'mcErrors']
modularAnalysis.variablesToNTuple(vertex_parent, rootVars, filename='sample.root', path=path)

basf2.process(path)
