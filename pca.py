import kfp
from kfp.dsl import component
from kfp.dsl import Output, Metrics
from typing import NamedTuple

@component(
    packages_to_install=[
        "scikit-learn==1.0.2",
        "pandas==1.3.5",
        "matplotlib==3.5.1",
        "numpy==1.23.0"
    ],
    base_image="python:3.9"
)
def perform_pca(
     Output[kfp.dsl.Dataset],
    metrics: Output[Metrics],
    n_components: int = 2
) -> NamedTuple("Outputs", [("explained_variance_ratio", str)]):
    """
    Performs PCA on the Iris dataset, saves the transformed data,
    and generates a scree plot.

    Args:
         Input dataset.
        metrics: Output metrics.
        n_components (int): The number of principal components to retain.

    Returns:
        explained_variance_ratio (str): The explained variance ratio.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    import os
    from collections import namedtuple

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    # Create a Pandas DataFrame for the transformed data
    df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])
    df['target'] = y  # Add the target variable for potential analysis

    # Save the transformed data to a CSV file
    output_path = os.path.join(data.path, "pca_transformed_data.csv")
    df.to_csv(output_path, index=False)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance)

    # Log metrics
    metrics.log_metric("explained_variance_ratio_PC1", explained_variance[0])
    if n_components > 1:
        metrics.log_metric("explained_variance_ratio_PC2", explained_variance[1])

    # Cumulative explained variance
    cum_var_exp = np.cumsum(explained_variance)

    # Scree plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.savefig(os.path.join(data.path, "scree_plot.png"))  # Save the plot to a file
    plt.close()

    metrics.log_artifact("scree_plot", data.path + "/scree_plot.png")

    # Return the explained variance ratio as a string
    Outputs = namedtuple('Outputs', ['explained_variance_ratio'])
    return Outputs(explained_variance_ratio=str(explained_variance.tolist()))
