# CNN Robustness to Camera Occlusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This research project investigates the robustness of Convolutional Neural Networks (CNNs) against common camera occlusion effects like rain and dust. Using the German Traffic Sign Recognition Benchmark (GTSRB) dataset, we train models under various conditions and analyze their performance degradation and internal representations.

## Key Features

* **Realistic Occlusion Effects:** Utilizes the `camera-occlusion` library to programmatically add rain and dust effects to training and test data.
* **Config-Driven Experiments:** The entire experimental workflow—from model architecture to training conditions and analysis parameters—is controlled via a central YAML configuration file.
* **Comprehensive Analysis:** Automatically generates a detailed markdown report with performance heatmaps, robustness metrics, and statistical significance tests.
* **Model Introspection Tools:** Provides command-line tools to visualize learned filters and feature map activations, allowing for deep inspection of model behavior under clean and noisy conditions.

## Project Structure

CODE_BREAK
cnn-occlusion-robustness/
├── configs/
│   └── eval/
│       └── matrix.yaml        # Main config for experiments
├── results/                   # Raw output (models, eval JSONs, matrices)
├── analysis_output/           # Final reports and publication-quality figures
├── scripts/
│   ├── run_all.sh             # Main script to run the full pipeline
│   └── organize_test_set.py   # One-time script to structure the GTSRB test set
├── src/
│   └── cnn_occlusion_robustness/
│       ├── analysis/          # Analysis and visualization scripts
│       ├── data/              # Dataset handling
│       ├── models/            # CNN architecture definitions
│       ├── train.py           # Training script
│       └── eval.py            # Evaluation script
├── pyproject.toml             # Project definition and dependencies
└── README.md
CODE_BREAK

## Setup and Installation

#### 1. Clone the Repository
CODE_BREAKbash
git clone [https://github.com/your-username/cnn-occlusion-robustness.git](https://github.com/your-username/cnn-occlusion-robustness.git)
cd cnn-occlusion-robustness
CODE_BREAK

#### 2. Create a Virtual Environment
It's recommended to use a virtual environment.
CODE_BREAKbash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
CODE_BREAK

#### 3. Install Dependencies
Install the project in editable mode (`-e`), which automatically discovers and installs all dependencies from `pyproject.toml` and registers the command-line tools.
CODE_BREAKbash
pip install -e .
CODE_BREAK

#### 4. Download and Prepare the Dataset
1.  Download the GTSRB dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce11523b6998da45b0c7961/GTSRB_Final_Training_Images.zip) (Training Set) and [here](https://sid.erda.dk/public/archives/daaeac0d7ce11523b6998da45b0c7961/GTSRB_Final_Test_Images.zip) (Test Set).
2.  Unzip them into a parent directory, for example `../GTSRB_dataset/`.
3.  The GTSRB test set images are provided in a flat folder. Run the provided script once to organize them into class-specific subfolders, which is required for evaluation.
CODE_BREAKbash
python scripts/organize_test_set.py
CODE_BREAK

## Running Experiments

#### 1. Configure Your Experiment
The entire experiment is controlled by `configs/eval/matrix.yaml`. Here you can define:
* `training_conditions`: A list of effects to train models on (e.g., `none`, `heavy_rain`).
* `test_conditions`: A list of effects to evaluate all trained models against.
* `model`: The CNN architecture and its hyperparameters (`kernel_size`, `channels`, etc.).
* Training parameters like `epochs` and `learning_rate`.

#### 2. Run the Full Pipeline
The easiest way to run a complete experiment is using the `run_all.sh` script.
CODE_BREAKbash
bash scripts/run_all.sh
CODE_BREAK
This script automates the four main phases of the experiment:
1.  **Training:** Trains a separate model for each condition in `training_conditions`.
2.  **Evaluation:** Evaluates every model against every condition in `test_conditions`.
3.  **Matrix Building:** Aggregates all evaluation results into a single CSV performance matrix.
4.  **Analysis:** Generates a comprehensive report, tables, and figures in the `analysis_output/` directory.

## Command-Line Tools

This project includes several command-line tools for running specific tasks or performing deeper analysis.

#### `analyze-results`
Generates the final analysis report from an existing set of results.
CODE_BREAKbash
analyze-results --results-dir ./results --output-dir ./analysis_output
CODE_BREAK

#### `visualize-filters`
Visualizes the learned kernels (filters) of a specific convolutional layer in a trained model. This is useful for seeing what low-level patterns the model has learned.

*Example: Visualize `conv1` filters from the `mixed_heavy` model:*
CODE_BREAKbash
visualize-filters \
    --config configs/eval/matrix.yaml \
    --model-path results/models/mixed_heavy_model.pth \
    --output-path conv1_filters.png
CODE_BREAK

*Example: Visualize `conv2` filters from the same model:*
CODE_BREAKbash
visualize-filters \
    --config configs/eval/matrix.yaml \
    --model-path results/models/mixed_heavy_model.pth \
    --layer-name conv2 \
    --output-path conv2_filters.png
CODE_BREAK

#### `visualize-activations`
Visualizes the feature map activations for a given input image. This shows how the model "sees" a specific image and which features are triggered by clean or occluded inputs.

*Example: See how the model activates on a clean test image:*
CODE_BREAKbash
visualize-activations \
    --config configs/eval/matrix.yaml \
    --model-path results/models/mixed_heavy_model.pth \
    --image-path ../GTSRB_dataset/GTSRB_test/Final_Test/Images/00000/00243.ppm \
    --output-dir activations_clean
CODE_BREAK

*Example: See activations for the same image but with a `heavy_dust` effect applied:*
CODE_BREAKbash
visualize-activations \
    --config configs/eval/matrix.yaml \
    --model-path results/models/mixed_heavy_model.pth \
    --image-path ../GTSRB_dataset/GTSRB_test/Final_Test/Images/00000/00243.ppm \
    --test-effect heavy_dust \
    --output-dir activations_heavy_dust
CODE_BREAK

## Understanding the Output

After a full run, the key outputs will be in:
* `results/`: Contains the trained models (`.pth`), individual evaluation scores (`.json`), and the final performance matrix (`.csv`).
* `analysis_output/`: Contains the final results.
    * `reports/comprehensive_analysis.md`: The main report with key findings.
    * `figures/performance_heatmap.png`: A visualization of the cross-domain performance matrix.
    * `tables/`: All calculated metrics in CSV format.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
