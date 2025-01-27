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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import mean_squared_error, r2_score

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

calc = Calculator(descriptors, ignore_3D=True)

train_file = data_dir / "train_winsorized.parquet"
# val_file = data_dir / "validation.parquet"
if not train_file.exists():
    train_data = pd.read_csv(data_dir / "AqSolDBc.csv")
    train_data["rdkit_mol"] = train_data["SmilesCurated"].apply(MolFromSmiles)
    train_data = train_data.dropna(axis=0, subset=["rdkit_mol"])
    # determine our own training and validation sets
    # *_, train_idxs, val_idxs = train_test_split_molecules(train_data["rdkit_mol"], sampler="kmeans", train_size=0.80, test_size=0.20, return_indices=True)
    train_descs: pd.DataFrame = calc.pandas(train_data["rdkit_mol"]).fill_missing()
    
    # winsorization
    train_descs = train_descs.astype(float, errors='ignore')
    feature_means = train_descs.mean(axis=0, skipna=True)
    feature_stdev = train_descs.var(axis=0, skipna=True).pow(0.5)
    n_sigma = 3
    train_descs.clip(lower=feature_means - n_sigma * feature_stdev, upper=feature_means + n_sigma * feature_stdev, axis=1, inplace=True)

    train_df = pd.concat((train_data[["ExperimentalLogS"]], train_descs), axis=1)
    train_df.to_parquet(train_file)
    # train_df.iloc[val_idxs].to_parquet(val_file)
train_df = pd.read_parquet(train_file)
# val_df = pd.read_parquet(val_file)


test_file = data_dir / "test_winsorized.parquet"
if not test_file.exists():
    test_data = pd.read_csv(data_dir / "OChemUnseen.csv")
    test_data["rdkit_mol"] = test_data["SMILES"].apply(MolFromSmiles)
    test_data = test_data.dropna(axis=0, subset=["rdkit_mol"])
    test_descs: pd.DataFrame = calc.pandas(test_data["rdkit_mol"]).fill_missing()
    
    # winsorization - note that we use the training statistics to avoid a data leak
    test_descs = test_descs.astype(float, errors='ignore')
    test_descs.clip(lower=feature_means - n_sigma * feature_stdev, upper=feature_means + n_sigma * feature_stdev, axis=1, inplace=True)
    
    test_df = pd.concat((test_data[["LogS"]], test_descs), axis=1)
    test_df.to_parquet(test_file)
test_df = pd.read_parquet(test_file)
# match the target column names
test_df = test_df.rename(columns={"LogS": "ExperimentalLogS"})

train_data = TabularDataset(train_df)
# val_data = TabularDataset(val_df)
test_data = TabularDataset(test_df)
'''
# training from scratch
outdir = '/datai/autogluon_random_winsorization'
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
    # tuning_data=val_data,
    # use_bag_holdout=True,
    test_data=test_data,  # not seen during training
    num_gpus=8,
    presets='best_quality',
    time_limit=3600*100,  # 100 hours
)
'''
# initial quick run
# predictor = TabularPredictor.load("/home/jwburns/autosol/AutogluonModels/ag-20250113_212151")
# predictor = TabularPredictor.load("/datai/autogluon")  # best_quality random splitting based model
predictor = TabularPredictor.load("/datai/autogluon_random_winsorization")  # winsorized, random split
# print(f"Test Set Performance:", predictor.evaluate(test_data))
predictions = predictor.predict(test_data, as_pandas=False)

data = pd.DataFrame(
    {
        "truth": test_data["ExperimentalLogS"],
        "prediction": predictions,
    }
)

# Create the plot
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax: Axes  # type hints, please
ax.grid(True, which="major", axis="both")
ax.set_axisbelow(True)
hb = ax.hexbin(x=data["truth"], y=data["prediction"], gridsize=70, cmap="viridis", mincnt=1)

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Number of compounds")
ax.plot([-12, 2], [-12, 2], "r", linewidth=1)
ax.plot([-12, 2], [-11, 3], "r--", linewidth=0.5)
ax.plot([-12, 2], [-13, 1], "r--", linewidth=0.5)
ax.set_title("autosol")
ax.set_xlabel("Solubility (LogS)")
ax.set_ylabel("cLogS")
ax.set_xlim(-12, 2)
ax.set_ylim(-12, 2)

# Text box for R2 and MSE
textstr = "\n".join(
    (
        f"$\\bf{{R2}}:$ {r2_score(data['truth'], data['prediction']):.2f}",
        f"$\\bf{{MSE}}:$ {mean_squared_error(data['truth'], data['prediction']):.2f}",
    )
)
ax.text(
    -8.55,
    -2.1,
    textstr,
    transform=ax.transData,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
)

# Pie chart
_frac_wn_1 = np.count_nonzero(np.abs(data["truth"] - data["prediction"]) < 1.0) / len(data)
sizes = [1 - _frac_wn_1, _frac_wn_1]
ax_inset = ax.inset_axes([-12, -2, 4, 4], transform=ax.transData)
ax_inset.pie(
    sizes,
    colors=["#ae2b27", "#4073b2"],
    startangle=360 * (_frac_wn_1 - 0.5) / 2,
    wedgeprops={"edgecolor": "black"},
    autopct="%1.f%%",
    textprops=dict(color="w"),
)
ax_inset.axis("equal")

plt.savefig("parity.png")
