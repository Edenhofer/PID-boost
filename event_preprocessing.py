import basf2
import modularAnalysis
import pdg

path = basf2.create_path()

# Base definitions of stable particles and detector data
particle = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

# Default list of variables which should be exported to the root file
root_vars = ['isSignal', 'mcErrors', 'mcPDG']
root_vars += ['kaonID', 'pionID','electronID',  'muonID', 'protonID', 'deuteronID']

# Import mdst file and fill particle list without applying any cuts
modularAnalysis.inputMdstList("default", ['sample.mdst.root'], path=path)
modularAnalysis.fillParticleLists([(p, '') for p in particle], path=path)

for p in particle:
    child_vars = []
    for d in detectors:
        for p_2 in particle:
            child_vars += ['pidLogLikelihoodValueExpert(' + str(pdg.from_name(p_2)) + ', ' + str(d) + ')']

    # Export variables of the analysis to NTuple root file
    # Inspect the value using modularAnalysis.printVariableValues('K+', `varName(varArg)`, path=path)
    modularAnalysis.matchMCTruth(p, path=path)
    modularAnalysis.variablesToNTuple(p, root_vars + child_vars, filename=p + '.root', path=path)

basf2.process(path)
