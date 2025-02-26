# MLOps Pipeline Project

This project contains two main components:

1.  **PCA Implementation (`PCA_test.ipynb`):** A Jupyter Notebook that demonstrates how to perform Principal Component Analysis (PCA) on the Iris dataset using Kubeflow Pipelines (KFP).
2.  **Census Data Pipeline (`kfp2_pipe_base.py`):** A Python script that defines a Kubeflow Pipeline for training and deploying an XGBoost model on census data.

## 1. PCA Implementation (`PCA_test.ipynb`)

### Overview

This notebook performs PCA on the Iris dataset and visualizes the results. It includes the following steps:

*   **Installation of Dependencies:** Installs necessary packages such as `google-cloud-aiplatform`, `google-cloud-storage`, `numpy`, and `google-cloud-pipeline-components`.
*   **Project Setup:** Configures the Google Cloud project ID, location, and Cloud Storage bucket URI.
*   **Service Account Configuration:** Sets up the service account for accessing Google Cloud resources.
*   **PCA Implementation:**
    *   Loads the Iris dataset.
    *   Standardizes the data.
    *   Performs PCA to reduce the dimensionality of the dataset.
    *   Saves the transformed data to a CSV file.
    *   Generates a scree plot to visualize the explained variance ratio.
*   **Pipeline Definition:** Defines a Kubeflow Pipeline that executes the PCA component.
*   **Pipeline Compilation and Execution:** Compiles the pipeline and submits it to Vertex AI Pipelines for execution.

### Components

*   **`perform_pca` Component:** This component performs the PCA, saves the transformed data, and generates the scree plot. It takes the number of components as input and outputs a dataset artifact containing the transformed data and metrics.

### How to Run

1.  Open the `PCA_test.ipynb` notebook in a Jupyter environment.
2.  Configure the `PROJECT_ID`, `LOCATION`, `BUCKET_URI`, and `SERVICE_ACCOUNT` variables.
3.  Run the notebook cells sequentially to execute the PCA pipeline.

## 2. Census Data Pipeline (`kfp2_pipe_base.py`)

### Overview

This script defines a Kubeflow Pipeline that automates the process of training and deploying an XGBoost model on census data. The pipeline includes the following steps:

*   **Setup:**
    *   Installs necessary packages, including `kfp`, `google-cloud-aiplatform`, `google-cloud-bigquery`, `xgboost`, `pandas`, `numpy`, `joblib`, and `scikit-learn`.
    *   Configures the Google Cloud project ID, location, and Cloud Storage bucket URI.
    *   Sets up the service account for accessing Google Cloud resources.
*   **Components:**
    *   **`create_census_view`:** Creates a BigQuery view on the public census dataset.
    *   **`export_dataset`:** Exports data from the BigQuery view to a CSV file.
    *   **`xgboost_training`:** Trains an XGBoost model using the exported dataset.
    *   **`deploy_xgboost_model`:** Deploys the trained XGBoost model to Vertex AI.
*   **Pipeline Definition:** Defines the Kubeflow Pipeline that orchestrates the execution of the components.
*   **Pipeline Execution:** Compiles the pipeline and submits it to Vertex AI Pipelines for execution.

### Components

*   **`create_census_view`:** Creates a BigQuery view on the public census dataset.
*   **`export_dataset`:** Exports data from the BigQuery view to a CSV file.
*   **`xgboost_training`:** Trains an XGBoost model using the exported dataset.
*   **`deploy_xgboost_model`:** Deploys the trained XGBoost model to Vertex AI.

### How to Run

1.  Ensure you have the Google Cloud SDK installed and configured.
2.  Set the `PROJECT_ID`, `LOCATION`, `BUCKET_URI`, and `SERVICE_ACCOUNT` variables in the script.
3.  Run the script to execute the pipeline:

    ```bash
    python kfp2_pipe_base.py
    ```

## Additional Information

*   Ensure that the Cloud Storage bucket and BigQuery dataset are created before running the pipelines.
*   Grant the necessary permissions to the service account to access Google Cloud resources.
*   Refer to the Kubeflow Pipelines and Vertex AI documentation for more information on pipeline development and execution.
