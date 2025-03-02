import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, MolSurf
from rdkit.Chem.EState import EState

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 20  # Return None if molecule cannot be parsed

    BCUT2D_MWLOW = Descriptors.BCUT2D_MWLOW(mol)
    SMR_VSA1 = Chem.MolSurf.SMR_VSA1(mol)
    MinEStateIndex = Chem.EState.EState.MinEStateIndex(mol)
    VSA_EState5 = rdkit.Chem.EState.EState_VSA.VSA_EState5(mol)
    VSA_EState6 = rdkit.Chem.EState.EState_VSA.VSA_EState6(mol)
    VSA_EState7 = rdkit.Chem.EState.EState_VSA.VSA_EState7(mol)
    PEOE_VSA14 = MolSurf.PEOE_VSA14(mol)
    MolLogP = Descriptors.MolLogP(mol)

    return [MinEStateIndex, SMR_VSA1, BCUT2D_MWLOW, VSA_EState5,
            VSA_EState6, VSA_EState7, PEOE_VSA14, MolLogP]

df = pd.read_csv('2_peptide.csv', keep_default_na=False)
df['SMILES'] = df['PepName'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromFASTA(x)))  # Convert peptide to SMILES
df_descriptors = df['SMILES'].apply(calculate_descriptors)

descriptors_df = pd.DataFrame(df_descriptors.tolist(), columns=[
    "MinEStateIndex", "SMR_VSA1", "BCUT2D_MWLOW", "VSA_EState5",
    "VSA_EState6", "VSA_EState7", "PEOE_VSA14", "MolLogP"
])

result_df = pd.concat([df, descriptors_df], axis=1)

# Save the results to a new CSV file
output_file = './input.csv'
result_df.to_csv(output_file, index=False)

print(f'Results saved to {output_file}')
