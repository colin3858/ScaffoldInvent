
# import warnings
# warnings.filterwarnings("ignore")
import os
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMMPA
from rdkit.Chem import Lipinski, Descriptors
from joblib import Parallel, delayed
from multiprocessing import Pool

def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-input_data_path", "-i", required=True,
                        help="SDF file of the molecules")

    parser.add_argument("-n_jobs", "-n", required=True,
                         help="number of the cores", type=int)

    parser.add_argument("-output", "-o", required=True,
                        help="Generated mmps pairs")


    return parser.parse_args()


def filter(mol, type = "frags"):

    HBD = Lipinski.NumHDonors(mol)
    HBA = Lipinski.NumHAcceptors(mol)
    rings = len(Chem.GetSymmSSSR(mol))
    MW = Chem.Descriptors.MolWt(mol)

    if type == "frags":
        action = (HBD <=8) & (HBA <=8) & (rings >= 1) & (MW <=800)
    else:
        action = (HBD <= 5) & (HBA <= 5) & (MW <= 500)

    return action


# MMPs cutting algorithm
def mmps_cutting(smi, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", dummy=True, filtering=True):
    """ MMPs function"""
    FMQs = []
    fmq = None
    mol = Chem.MolFromSmiles(smi)
    try:
        # smi = Chem.MolToSmiles(mol)
        bricks = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, \
                                    pattern=pattern, resultsAsMols=False)

        for linker, chains in bricks:

            linker_mol = Chem.MolFromSmiles(linker)
            linker_size = linker_mol.GetNumHeavyAtoms()
            linker_site_idxs = [atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
            linker_length = len(Chem.rdmolops.GetShortestPath(linker_mol, \
                                                              linker_site_idxs[0], linker_site_idxs[1])) - 2

            if (linker_size >= 2) & (linker_length >1):
                frag1_mol = Chem.MolFromSmiles(chains.split(".")[0])
                frag2_mol = Chem.MolFromSmiles(chains.split(".")[1])
                frag1_size = frag1_mol.GetNumHeavyAtoms()
                frag2_size = frag2_mol.GetNumHeavyAtoms()


                if (frag1_size >= 5) & ((frag2_size >= 5) & ((frag1_size + frag1_size) >= linker_size)):

                    if filtering:

                        action = filter(linker_mol, type="frags") & filter(frag1_mol, type="frags") \
                                 & filter(frag2_mol, type="frags")
                        if action:

                            if dummy:
                                fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + ">" + "%s" % (smi)
                            else:
                                fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                      + "%s" % (remove_dummys(chains)) + ">" + "%s" % (smi)
                    else:

                        if dummy:
                            fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                  + "%s" % (chains) + ">" + "%s" % (smi)
                        else:
                            fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                  + "%s" % (remove_dummys(chains)) + ">" + "%s" % (smi)

                    FMQs.append(fmq)
    except:
        print("error")
        FMQs = []

    return FMQs


# remove dummy atoms(*) from MOL/SMILES format
def remove_dummys(smi_string):
    return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                                    Chem.MolFromSmiles('*'), \
                                                                    Chem.MolFromSmiles('[H]'), True)[0]))


def remove_dummys_mol(smi_string):
    return Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                   Chem.MolFromSmiles('*'), \
                                                   Chem.MolFromSmiles('[H]'), True)[0])


# fmq (dummy) example: L_2.C(C[*:2])[*:1].c1ccc([*:1])cc1.c1ccc([*:2])nc1 >c1ccccc1CCc2ccccn2

def main():
    opt = parse_args()
    mols = Chem.SDMolSupplier(opt.input_data_path)
    fmqs = Parallel(n_jobs=opt.n_jobs)(delayed(mmps_cutting)(i) for i in mols)

    fmqs = [j for i in fmqs for j in i if j]
    fmqs = list(set(fmqs))
    #
    w = open(opt.output, "w")
    for fmq in fmqs:
        w.write(fmq)
        w.write("\n")
    w.close()

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA

def mmps_cutting_from_csv(csv_file, output_file, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", dummy=True, filtering=True):
    """ MMPs function from CSV file"""
    data = pd.read_csv(csv_file, sep=';')
    # data = pd.read_csv(csv_file)
    smi=data['Smiles'].to_list()
    smi = [str(smile) for smile in smi if isinstance(smile, str)]
    # print(smi[10])
    fmqs=Parallel(n_jobs=32)(delayed(mmps_cutting)(i) for i in smi)
    fmqs = [j for i in fmqs for j in i if j]
    fmqs = list(set(fmqs))
    result_df = pd.DataFrame(fmqs)
    result_df.to_csv(output_file, index=False)
    # for idx, row in data.iterrows():
    #     smi = row['SMILES']
    #     fmqs=Parallel(n_jobs=2)(delayed(mmps_cutting)(i) for i in smi)
    #     fmqs = [j for i in fmqs for j in i if j]
    #     fmqs = list(set(fmqs))
    #     # fmqs = mmps_cutting(smi, pattern=pattern, dummy=dummy, filtering=filtering)
    #     if fmqs:  # Check if FMQs is not empty
    #         for fmq in fmqs:
    #             result_row = row.copy()  # Make a copy of the row
    #             result_row['FMQs'] = fmq  # Assign FMQ to 'FMQs' column
    #             results.append(result_row)  # Append the modified row
        
    # # Write results to output file
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(output_file, index=False)

# Example usage:
csv_file = '/home/lianhy/ScaffoldGVAE-master/data/mmp_test.csv'
output_file = '/home/lianhy/ScaffoldGVAE-master/data/chembl_brics_test.csv'
mmps_cutting_from_csv(csv_file, output_file)


