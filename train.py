import h2o
import mlflow
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
import pandas as pd

# Create dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

# Save dataset
df.to_csv("iris.csv",index=False)

print("Dataset created.")

# Start H2O
h2o.init()

# Import dataset into H2O
data = h2o.import_file("iris.csv")

# Train/Test split
train,test = data.split_frame(ratios=[0.8])

# Features + Target
x = data.columns[:-1]
y = "species"

# MLflow experiment
mlflow.set_experiment("H2O_AutoML_Experiment")

# AutoML
aml = H2OAutoML(
    max_models=5,
    seed=123
)

aml.train(
    x=x,
    y=y,
    training_frame=train
)

print("\nBEST MODEL\n")
print(aml.leader)

# Save model
h2o.save_model(
    aml.leader,
    path=".",
    force=True
)

print("\nTraining completed successfully.")