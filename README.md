# MLOps Pipeline Project

This project contains two main components:

1.  **PCA Implementation (`PCA_test.ipynb`):** A Jupyter Notebook that demonstrates how to perform Principal Component Analysis (PCA) on the Iris dataset using Kubeflow Pipelines (KFP).
2.  **Census Data Pipeline (`kfp2_pipe.ipynb`):** A Jupyter Notebook that defines a Kubeflow Pipeline for training and deploying an XGBoost model on census data.

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

## 2. Census Data Pipeline (`kfp2_pipe.ipynb`)

### Overview

This notebook defines a Kubeflow Pipeline that automates the process of training and deploying an XGBoost model on census data. It includes the following steps:

*   **Installation of Dependencies:** Installs necessary packages, including `kfp`, `google-cloud-aiplatform`, and others.
*   **Google Cloud Project Setup:** Configures the Google Cloud project ID, location, and Cloud Storage bucket URI.
*   **BigQuery Dataset Creation:** Creates a BigQuery dataset for storing the census data view.
*   **Component Definitions:** Defines the pipeline components for creating the BigQuery view, exporting the dataset, training the XGBoost model, and deploying the model.
*   **Pipeline Definition:** Defines the Kubeflow Pipeline that orchestrates the execution of the components.
*   **Pipeline Compilation and Execution:** Compiles the pipeline and submits it to Vertex AI Pipelines for execution.

### Components

*   **`create_census_view`:** Creates a BigQuery view on the public census dataset.
*   **`export_dataset`:** Exports data from the BigQuery view to a CSV file.
*   **`xgboost_training`:** Trains an XGBoost model using the exported dataset.
*   **`deploy_xgboost_model`:** Deploys the trained XGBoost model to Vertex AI.

### How to Run

1.  Open the `kfp2_pipe.ipynb` notebook in a Jupyter environment.
2.  Configure the `PROJECT_ID`, `LOCATION`, `BUCKET_URI`, and `SERVICE_ACCOUNT` variables.
3.  Run the notebook cells sequentially to execute the pipeline.

## Additional Information

*   Ensure that the Cloud Storage bucket and BigQuery dataset are created before running the pipelines.
*   Grant the necessary permissions to the service account to access Google Cloud resources.
*   Refer to the Kubeflow Pipelines and Vertex AI documentation for more information on pipeline development and execution.
