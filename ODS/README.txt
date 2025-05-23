Authors
Ervin Galas (34276705), Sofia Peeva (35133522), Eren Stannard (34189185)
ICT304: AI System Design
Murdoch University

Smart Occupancy Monitoring System

Overview
This project implements the full Smart Occupancy Monitoring System.
The system can detect the number of people in images using an Autoencoder AI technique.

Project Structure
- dataset/ - Dataset with varying levels of occupancy, consists of multiple datasets
- models/
  - autoencoder_model.py - Autoencoder model implementation
- config.py - Configuration parameters
- data_loader.py - Dataset loading utilities
- evaluate.py - Model evaluation utilities
- main.py - Main entry point
- predict.py - Prediction functionality
- train.py - Training script with progress indicators
- requirements.txt - Project dependencies
- README.txt - Project documentation (this file)

Dataset Structure
The system expects the dataset to be organised as follows:
- dataset/
  - frames/ - Contains image files (1.jpg, 2.jpg, etc.)
  - labels.csv - CSV file with count values for each image

The dataset incorporates a number of synthetic data samples to provide examples of environments
with zero occupancy, which is important for training the model to recognise empty spaces.

Requirements
- Python 3.7+
- PyTorch 1.9+
- Other dependencies listed in requirements.txt

Installation
1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt


## Usage

### Training
To train the Autoencoder model:

python main.py --mode train --model_type autoencoder


#### Additional training parameters:

--batch_size SIZE    - Batch size for training (default: 32)
--epochs NUM         - Number of training epochs (default: 20)
--lr RATE            - Learning rate (default: 0.001)


### Prediction
To predict occupancy from an image:

python main.py --mode predict --model_path output/autoencoder_model.pth --image path/to/image.jpg


To predict occupancy from camera feed:

python main.py --mode predict --model_path output/autoencoder_model.pth --camera

### Model Evaluation
To evaluate the model:

python evaluate.py --model_path output/autoencoder_model.pth

Additional evaluation parameters:

--data_dir PATH      - Path to test data directory
--labels_file PATH   - Path to test labels file
--batch_size SIZE    - Batch size for evaluation
--output_dir DIR     - Directory to save evaluation results

This will generate performance metrics (MSE, RMSE, MAE, R²) and a visualisation plot for the model.

## Model Details

### Autoencoder Model
Uses an encoder-decoder architecture to extract features and predict occupancy.

## Dataset Details

### Mall Dataset (archive)
Contains 2000 images from mall surveillance with varying levels of occupancy. Each image has a
corresponding count label in the labels.csv file.

### Human Crowd Detection Dataset (NewData)
Contains images with varying crowd densities. Each image has a corresponding text file with
bounding box coordinates for each person.

### Synthetic No People Dataset (synthetic_no_people)
Contains images of indoor environments with no people. This dataset is crucial for training the
model to recognize empty spaces, providing examples with zero occupancy.

## Output
- Trained models are saved in the 'output' directory by default.
- Evaluation results, including performance metrics and plots, are saved in the output directory.
- Prediction results display integers since fractional people don't make sense in occupancy context.

## Troubleshooting
- If no output appears during training, ensure that tqdm is installed: `pip install tqdm`
- If dataset loading fails, check that the data directory structure matches the expected format
- For CUDA errors, ensure your PyTorch installation matches your GPU driver version