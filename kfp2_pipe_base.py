#!/usr/bin/env python
# coding: utf-8


get_ipython().system(' pip3 install --no-cache-dir --upgrade "kfp>2"                                          google-cloud-aiplatform')


# Check the KFP SDK version.

# In[3]:


get_ipython().system(' python3 -c "import kfp; print(\'KFP SDK version: {}\'.format(kfp.__version__))"')
get_ipython().system(' pip3 freeze | grep aiplatform')




import sys



PROJECT_ID = get_ipython().getoutput('(gcloud config get-value core/project)')
PROJECT_ID = PROJECT_ID[0]

LOCATION = "us-central1"
LOCATION = "us-central1"
BQ_LOCATION = LOCATION.split("-")[0].upper()


# ### Create a Cloud Storage bucket
# 
# Create a storage bucket to store intermediate artifacts such as datasets.

# In[7]:


BUCKET_URI = f"gs://{PROJECT_ID}-unique"  # @param {type:"string"}


# **Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.



get_ipython().system(' gsutil mb -l $LOCATION -p $PROJECT_ID $BUCKET_URI')


# #### Service Account 



SERVICE_ACCOUNT = "[your-service-account]"  # @param {type:"string"}


# In[10]:


import sys

IS_COLAB = "google.colab" in sys.modules
if (
    SERVICE_ACCOUNT == ""
    or SERVICE_ACCOUNT is None
    or SERVICE_ACCOUNT == "[your-service-account]"
):
    # Get your service account from gcloud
    if not IS_COLAB:
        shell_output = get_ipython().getoutput('gcloud auth list 2>/dev/null')
        SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()

    else:  # IS_COLAB:
        shell_output = get_ipython().getoutput(' gcloud projects describe  $PROJECT_ID')
        project_number = shell_output[-1].split(":")[1].strip().replace("'", "")
        SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"

    print("Service Account:", SERVICE_ACCOUNT)


# #### Set service account access for Vertex AI Pipelines
# 


get_ipython().system(' gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI')

get_ipython().system(' gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI')


# ### Import libraries and define constants

# In[12]:


import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component


# ## Initialize Vertex AI SDK for Python
# 
# Initialize the Vertex AI SDK for Python for your project and corresponding bucket.

# In[13]:


aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)


# In[14]:


PATH = get_ipython().run_line_magic('env', 'PATH')
get_ipython().run_line_magic('env', 'PATH={PATH}:/home/jupyter/.local/bin')


DATASET_ID = "census"  # The Data Set ID where the view sits
VIEW_NAME = "census_data"  # BigQuery view you create for input data

# KFP_ENDPOINT = (
#     "https://720c5bc00c3d6089-dot-us-central1.pipelines.googleusercontent.com/"
# )

PIPELINE_ROOT = f"{BUCKET_URI}/census_pipeline"  # This is where all pipeline artifacts are sent. You'll need to ensure the bucket is created ahead of time
PIPELINE_ROOT
print(f"PIPELINE_ROOT: {PIPELINE_ROOT}")


# ### Create a BigQuery dataset
# 


# Create a BQ Dataset in the project.
get_ipython().system('bq mk --location=$BQ_LOCATION --dataset $PROJECT_ID:$DATASET_ID')


# ## Define components for the pipeline
# 



@component(
    packages_to_install=["google-cloud-bigquery==3.10.0"],
)
def create_census_view(
    project_id: str,
    dataset_id: str,
    view_name: str,
):
    """Creates a BigQuery view on `bigquery-public-data.ml_datasets.census_adult_income`.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    create_or_replace_view = """
        CREATE OR REPLACE VIEW
        `{dataset_id}`.`{view_name}` AS
        SELECT
          age,
          workclass,
          education,
          education_num,
          marital_status,
          occupation,
          relationship,
          race,
          sex,
          capital_gain,
          capital_loss,
          hours_per_week,
          native_country,
          income_bracket,
        FROM
          `bigquery-public-data.ml_datasets.census_adult_income`
    """.format(
        dataset_id=dataset_id, view_name=view_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=create_or_replace_view, job_config=job_config)
    query_job.result()


# ### Define export dataset component
# 
# Next, you define a component to export the data from your BigQuery dataset to use for training the model.

# In[17]:


@component(
    packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def export_dataset(
    project_id: str,
    dataset_id: str,
    view_name: str,
    dataset: Output[Dataset],
):
    """Exports from BigQuery to a CSV file.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.

    Returns:
        dataset: The Dataset artifact with exported CSV file.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table_name = f"{project_id}.{dataset_id}.{view_name}"
    query = """
    SELECT
      *
    FROM
      `{table_name}`
    """.format(
        table_name=table_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)


# ### Define XGBoost training component
# 
# Next, you define a component to train an XGBoost model with the training data.

# In[18]:


@component(
    packages_to_install=[
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
    ],
)
def xgboost_training(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains an XGBoost classifier.

    Args:
        dataset: The training dataset.

    Returns:
        model: The model artifact stores the model.joblib file.
        metrics: The metrics of the trained model.
    """
    import os

    import joblib
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder

    # Load the training census dataset
    with open(dataset.path, "r") as train_data:
        raw_data = pd.read_csv(train_data)

    CATEGORICAL_COLUMNS = (
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    )
    LABEL_COLUMN = "income_bracket"
    POSITIVE_VALUE = " >50K"

    # Convert data in categorical columns to numerical values
    encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        raw_data[col] = encoders[col].fit_transform(raw_data[col])

    X = raw_data.drop([LABEL_COLUMN], axis=1).values
    y = raw_data[LABEL_COLUMN] == POSITIVE_VALUE

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    _ = xgb.DMatrix(X_train, label=y_train)
    _ = xgb.DMatrix(X_test, label=y_test)

    params = {
        "reg_lambda": [0, 1],
        "gamma": [1, 1.5, 2, 2.5, 3],
        "max_depth": [2, 3, 4, 5, 10, 20],
        "learning_rate": [0.1, 0.01],
    }

    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        objective="binary:hinge",
        silent=True,
        nthread=1,
        eval_metric="auc",
    )

    folds = 5
    param_comb = 20

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=params,
        n_iter=param_comb,
        scoring="precision",
        n_jobs=4,
        cv=skf.split(X_train, y_train),
        verbose=4,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    xgb_model_best = random_search.best_estimator_
    predictions = xgb_model_best.predict(X_test)
    score = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    _ = precision_recall_curve(y_test, predictions)

    metrics.log_metric("accuracy", (score * 100.0))
    metrics.log_metric("framework", "xgboost")
    metrics.log_metric("dataset_size", len(raw_data))
    metrics.log_metric("AUC", auc)

    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(xgb_model_best, os.path.join(model.path, "model.joblib"))


# ### Define deploying the model component
# 
# Finally, you define a component to deploy the XGBoost model.

# In[19]:


@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_xgboost_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="census-demo-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


# ## Construct the XGBoost training pipeline
# 
# Now you define the pipeline, with the following steps:
# 
# - Create a BigQuery view of the dataset.
# - Export the dataset.
# - Train the model.
# - Deploy the model.

# In[20]:


@dsl.pipeline(
    name="census-demo-pipeline",
)
def pipeline():
    """A demo pipeline."""

    create_input_view_task = create_census_view(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        view_name=VIEW_NAME,
    )

    export_dataset_task = (
        export_dataset(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            view_name=VIEW_NAME,
        )
        .after(create_input_view_task)
        .set_caching_options(False)
    )

    training_task = xgboost_training(
        dataset=export_dataset_task.outputs["dataset"],
    )

    _ = deploy_xgboost_model(
        project_id=PROJECT_ID,
        model=training_task.outputs["model"],
    )


# ### Compile the pipeline
# 
# Next, you compile the pipeline. 

# In[21]:


compiler.Compiler().compile(pipeline_func=pipeline, package_path="pipeline.yaml")


# ### Run the pipeline using Vertex AI Pipelines
# 
# Now, run the compiled pipeline.

# In[22]:


job = aiplatform.PipelineJob(
    display_name="census-demo-pipeline",
    template_path="pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
)

job.run()


# ### Run the pipeline using KFP
# 
# Alternatively, you can run the pipeline using KFP directly.
# 
# *Note:* You need to provide your own value for `KFP_ENDPOINT`
