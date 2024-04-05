# Fed Neuralnet Microbiome 

## Description
This featurecloud app trains a federated machine learning model to detect colorectal cancer (CRC) based on microbiota.
It can also be used to train a model to train any other label.

## Inputs
The app requires the following inputs to start and execute properly:

1. **`config.yml` Configuration File**:
   - **Format**: YAML
   - **Description**: A YAML file containing configuration parameters for the application, including paths to input data, model parameters, and execution settings.
   - **Contents**:
    ```yaml
    microbiome_net_learner:
        data: "PRJEB6070-data.csv"  # Path to the CSV file containing the dataset
        target: "host_phenotype"    # The name of the target column in the dataset
        sep: ","                    # the seperator used in the csvv file
        max_iterations: 10           # Maximum number of iterations to run the computation (default: 10)
        epochs_per_iteration: 10    # Number of epochs for each iteration of model training (default: 10)
        batch_size: 26              # Batch size for training the model (default: 26)
        learning_rate: 0.001        # Learning rate for the optimizer (default: 0.001).
        test_size: 0.1              # Proportion of the dataset to be used as validation data (default: 0.1)
    ```
   - **Preparation**:
     - Create a `config.yml` file in the specified input directory.
     - Add the required configuration parameters according to the application's needs.

2. **Data File**:
   - **Format**: CSV
   - **Description**: A CSV file containing the dataset to be used for training and validation.
   - **Preparation**:
     - Ensure the dataset is in CSV format with a header row specifying column names.
     - Place the CSV file in the location specified by the `data` parameter in `config.yml`.

## Outputs

The app generates several outputs as a result of its execution, which are saved to the specified output directory:

1. **Model File** (`model.pt`):
   - **Description**: A PyTorch model file containing the trained model's state dictionary.
   - **Usage**: Can be loaded into a PyTorch model object for inference or further training.

2. **Training History** (`history.json`):
   - **Format**: JSON
   - **Description**: A JSON file containing the training history, including metrics such as loss, accuracy, and AUC for each epoch.
   - **Usage**: Can be used for analysis of the training process, including plotting performance metrics over time.

3. **Performance Plots** (`history.png`):
   - **Format**: PNG image
   - **Description**: A set of plots derived from the training history, showing the progression of loss, accuracy, and AUC metrics through the epochs.
   - **Usage**: Provides a visual representation of the model's training performance.

The files can be found in a `results` folder that will be created by the app.

## Example data
Some example data is provided in the GitHub repository in the data folder. This example data is taken
from [\[1\]](https://doi.org/10.57745/7IVO3E).

## Additional models
Additionally to the feed forward model provided in the app, the source code contains
other models that could be applied locally. These models can be found in `src/central_model_feedForward.py`.
A simple convolutional network as well as a ResNet is provided.

# References
[1] BARBET, P., ALMEIDA, M., PROBUL, N., BAUMBACH, J., PONS, N., PLAZA ONATE, F., & LE CHATELIER, E. (2022). Taxonomic profiles, functional profiles and manually curated metadata of human fecal metagenomes from public projects coming from colorectal cancer studies (Version V8) [Computer software]. Recherche Data Gouv. https://doi.org/10.57745/7IVO3E 