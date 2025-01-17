"""
main.py

Driver for script for AutoML on solubility data

Notes:

1/17/25
initial run yielded a dramatically overfit predictor, 5x worse on test than val.
suggests that we need to preemptively split the data into training and validation
in such a way as to reflect the task of moving to the test set (extrapolating).
"""
from pathlib import Path

from astartes.molecules import train_test_split_molecules
from autogluon.tabular import TabularDataset, TabularPredictor
from mordred import Calculator, descriptors
import pandas as pd
from rdkit.Chem import MolFromSmiles

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

calc = Calculator(descriptors, ignore_3D=True)

train_file = data_dir / "train.parquet"
val_file = data_dir / "validation.parquet"
if not train_file.exists():
    train_data = pd.read_csv(data_dir / "AqSolDBc.csv")
    train_data["rdkit_mol"] = train_data["SmilesCurated"].apply(MolFromSmiles)
    train_data = train_data.dropna(axis=0, subset=["rdkit_mol"])
    # determine our own training and validation sets
    *_, train_idxs, val_idxs = train_test_split_molecules(train_data["rdkit_mol"], sampler="kmeans", train_size=0.80, test_size=0.20, return_indices=True)
    train_descs: pd.DataFrame = calc.pandas(train_data["rdkit_mol"]).fill_missing()
    train_df = pd.concat((train_data[["ExperimentalLogS"]], train_descs), axis=1)
    train_df.iloc[train_idxs].to_parquet(train_file)
    train_df.iloc[val_idxs].to_parquet(val_file)
train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)

test_file = data_dir / "test.parquet"
if not test_file.exists():
    test_data = pd.read_csv(data_dir / "OChemUnseen.csv")
    test_data["rdkit_mol"] = test_data["SMILES"].apply(MolFromSmiles)
    test_data = test_data.dropna(axis=0, subset=["rdkit_mol"])
    test_descs: pd.DataFrame = calc.pandas(test_data["rdkit_mol"]).fill_missing()
    test_df = pd.concat((test_data[["LogS"]], test_descs), axis=1)
    test_df.to_parquet(test_file)
test_df = pd.read_parquet(test_file)
# match the target column names
test_df = test_df.rename(columns={"LogS": "ExperimentalLogS"})

train_data = TabularDataset(train_df)
val_data = TabularDataset(val_df)
test_data = TabularDataset(test_df)

# training from scratch
outdir = '/datai/autogluon_extrapolation'
if Path(outdir).exists():
    print("Output dir already exists, exiting to avoid overwrite!")
    exit(1)
predictor = TabularPredictor(
    label="ExperimentalLogS",
    log_to_file=True,
    eval_metric='mean_squared_error',
    path=outdir,
)
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    use_bag_holdout=True,
    test_data=test_data,  # not seen during training
    num_gpus=8,
    presets='best_quality',
    time_limit=3600*100,  # 100 hours
)
# initial quick run
# predictor = TabularPredictor.load("/home/jwburns/autosol/AutogluonModels/ag-20250113_212151")

print(f"Test Set Performance:", predictor.evaluate(test_data))
print(predictor.leaderboard(test_data))
