# SGRA2500-ML: Spatial Guidance Refinement Adapter for Text-to-Image Diffusion Models

## Overview

**SGRA2500-ML** is a research-oriented machine learning project that explores spatial embedding refinement for text-to-image diffusion models through a custom **Spatial Guidance Refinement Adapter (SGRA)**.

The project investigates how text embeddings representing spatial relationships, such as *left of*, *behind*, *above*, and *beside*, can be refined using timestep-conditioned attention and a multi-objective loss function.

The implementation combines a pretrained T5 text encoder, a custom neural adapter, and a Stable Diffusion 1.5 pipeline to establish a prototype workflow for studying spatially structured text representations.

The project also includes an image-data preparation pipeline, embedding precomputation, checkpoint-based training, and a baseline-versus-SGRA embedding evaluation routine.

## Objectives

- Explore spatial relationship representation in text embeddings for text-to-image generation.
- Develop a timestep-conditioned neural adapter for refining text embeddings.
- Construct a dataset of 2,500 candidate images from the LAION-COCO aesthetic dataset, subject to filtering and download availability.
- Generate structured spatial captions using spaCy-based noun extraction and predefined spatial-relation pairs.
- Train the adapter using orthogonality, variance, and identity loss components.
- Compare baseline and refined embeddings using custom evaluation metrics.
- Save model checkpoints, evaluation tables, and visualization outputs for further analysis.

## Architecture

The implementation follows the workflow below:

1. **Dataset preparation:** Retrieve LAION-COCO metadata and select images with an aesthetic score of at least 6.0 and width and height of at least 512 pixels.
2. **Image preprocessing:** Download available images, convert them to RGB, resize them to 512 × 512 pixels, and save them with associated metadata.
3. **Spatial caption generation:** Extract nouns from existing captions using spaCy and combine them with predefined spatial-relation pairs.
4. **Text embedding extraction:** Encode the generated captions using the pretrained `google/t5-v1_1-base` text encoder and save the resulting tensors.
5. **SGRA adapter:** Apply timestep-conditioned multi-head attention, feed-forward refinement layers, residual connections, a learnable gate, and final normalization.
6. **Adapter training:** Optimize the adapter using a weighted combination of three loss components.
7. **Embedding evaluation:** Compare baseline and refined embeddings using custom pairwise, variance-based, and norm-based metrics.
8. **Result persistence:** Save model checkpoints, training curves, CSV reports, and LaTeX tables to Google Drive.

## Model Architecture

### 1. Pretrained Text Encoder

The project uses the `google/t5-v1_1-base` encoder to obtain contextual representations of spatial captions.

The encoded representations are stored as PyTorch tensors so they can be reused during adapter training without repeatedly running the text encoder.

### 2. SGRA Refinement Block

The `SGRA_Refinement_Block` class implements the core refinement operation.

Its main components are:

- **Multi-head attention:** Uses eight attention heads to refine text-token representations with timestep-derived features.
- **Layer normalization:** Normalizes intermediate representations.
- **Feed-forward network:** Applies a two-layer multilayer perceptron with GELU activation.
- **Residual connections:** Preserve a direct path from the original input.
- **Learnable gate:** Initializes at zero to control the initial contribution of the refinement block.

### 3. SGRA Adapter

The `SGRA_Adapter` class combines three refinement blocks with a timestep embedding network and a final projection and normalization layer.

| Component | Configuration |
|---|---|
| Embedding dimension | 768 |
| Attention heads | 8 |
| Refinement depth | 3 blocks |
| Feed-forward hidden dimension | 1,536 |
| Training batch size | 16 |
| Learning rate | 0.00005 |
| Training epochs | 5 |
| Optimizer | AdamW |
| Gradient clipping | Maximum norm of 1.0 |

These values reflect the configuration defined in the provided implementation.

## Multi-Objective Loss Function

The training objective combines three loss components to encourage changes in the embeddings while retaining aspects of their original representations.

### Orthogonality Loss

Penalizes the absolute mean dot product between the embedding refinement and the original embedding:

\[
L_{\mathrm{ortho}} =
\left|\operatorname{mean}(\Delta \odot E)\right|
\]

where \(E\) denotes the original embedding and \(\Delta\) denotes the difference between the refined and original embeddings.

### Variance Loss

Encourages the refined embeddings to approach a target variance equal to 1.4 times the original variance:

\[
L_{\mathrm{var}} =
\operatorname{MSE}\left(
\operatorname{Var}(E'),\,
1.4\operatorname{Var}(E)
\right)
\]

### Identity Loss

Uses cosine similarity to penalize changes in the direction of the original token representations:

\[
L_{\mathrm{id}} =
1-\operatorname{mean}\big(
\cos(E',E)
\big)
\]

### Combined Objective

The implemented objective is:

\[
L =
0.4L_{\mathrm{ortho}}
+0.4L_{\mathrm{var}}
+0.2L_{\mathrm{id}}
\]

Here, \(E'\) denotes the refined embedding. The three components are combined with weights of 0.4, 0.4, and 0.2, respectively.

## Dataset Preparation

The project retrieves metadata from the LAION-COCO aesthetic dataset hosted on Hugging Face.

The implemented filtering conditions are:

- Aesthetic score greater than or equal to 6.0.
- Image width greater than or equal to 512 pixels.
- Image height greater than or equal to 512 pixels.

Downloaded images are converted to RGB, resized to 512 × 512 pixels, and saved as JPEG files with quality set to 90.

The workflow maintains a JSON checkpoint containing image IDs, image paths, original captions, and source URLs. This allows the script to resume collection using previously recorded samples.

**Note:** The target is 2,500 images, but the number successfully downloaded depends on the availability and accessibility of the source images.

## Installation and Execution

### Recommended Environment

The notebook uses Google Colab and Google Drive for persistent storage. A CUDA-enabled GPU is recommended for model loading and adapter training.

### 1. Clone the Repository

```bash
git clone https://github.com/LaibaImran1500/SGRA2500-ML.git
cd SGRA2500-ML
```

### 2. Open the Notebook in Google Colab

Open the project notebook in Google Colab or upload the notebook to your Colab environment.

The supplied Python script contains notebook-specific commands, including shell commands prefixed with `!`, and is designed to run in a notebook environment rather than as an ordinary standalone Python script without modification.

### 3. Install Dependencies

The notebook installs the following packages:

```python
!pip install pyarrow huggingface_hub fastparquet transformers bitsandbytes accelerate diffusers spacy safetensors
!pip install seaborn matplotlib pandas numpy pillow requests tqdm
```

Install PyTorch and its CUDA-compatible dependencies according to the requirements of your runtime.

### 4. Mount Google Drive

Run the notebook's Drive mounting cell and authorize access when prompted.

The project uses the following base directory:

```text
/content/drive/MyDrive/SGRA_Project/
```

The directory stores the image dataset, metadata, embeddings, model checkpoints, and evaluation results.

### 5. Prepare the Dataset

Execute the image collection and preprocessing cells to create the image directory and checkpoint metadata.

The expected directory structure is:

```text
SGRA_Project/
├── images/
├── checkpoint_images.json
└── sgra_final_train_data.json
```

### 6. Generate Spatial Captions and Embeddings

Run the spaCy caption-processing cells, followed by the T5 embedding extraction cells.

The generated embeddings are stored under:

```text
SGRA_Project/
└── embeddings/
    ├── sgra_0000.pt
    ├── sgra_0001.pt
    └── ...
```

### 7. Train the SGRA Adapter

Execute the model definition, dataset, loss function, and training-loop cells.

The implementation trains for five epochs when a checkpoint is not found. The checkpoint is saved as:

```text
SGRA_Project/sgra_research_v1.pth
```

A training-loss plot is also generated:

```text
SGRA_Project/sgra_training_curve.png
```

If a checkpoint already exists, the notebook loads its weights and skips the initial training run.

### 8. Evaluate the Embeddings

Execute the inference and final audit cells to compare baseline and SGRA-refined text embeddings.

The evaluation routine also saves timestamped CSV and LaTeX reports under:

```text
SGRA_Project/results/
```

## Inference and Evaluation

The `sgra_inference()` function accepts a text prompt and an optional timestep value. It computes baseline T5 embeddings, applies the SGRA adapter, rescales the embedding adjustment, and clamps the refined output.

The final audit uses three example categories:

- **Spatial:** A square on the left and a circle on the right.
- **Shape:** A triangular plate beside a square napkin.
- **Complex:** A small yellow cube behind a large transparent box.

The following custom metrics are calculated:

| Metric | Purpose in the implementation |
|---|---|
| Pairwise Acc | Estimates separation between two averaged embedding segments using cosine similarity. |
| SR-Score | Tracks the overall embedding variance, scaled by 100. |
| VQAScore | Tracks the embedding norm, scaled by 1/100. |

The audit calculates baseline values, SGRA values, and percentage changes for each metric.

These are **custom embedding-based proxy metrics**, not independently validated measures of spatial accuracy, visual quality, or visual question-answering performance. The evaluation routine does not itself generate images or verify object positions in generated images.

## Outputs

The primary outputs include:

| Output | Description |
|---|---|
| `images/` | Downloaded and preprocessed images. |
| `checkpoint_images.json` | Image metadata and download progress. |
| `sgra_final_train_data.json` | Image references and generated spatial captions. |
| `embeddings/` | Precomputed PyTorch text-embedding tensors. |
| `sgra_research_v1.pth` | Trained adapter state dictionary. |
| `sgra_training_curve.png` | Training-loss visualization. |
| `SGRA_Audit_<timestamp>.csv` | Tabular baseline-versus-SGRA metrics. |
| `SGRA_Audit_<timestamp>.tex` | LaTeX version of the evaluation table. |

## Limitations and Future Work

The current implementation is a prototype for embedding refinement and requires further validation.

Potential extensions include:

- Integrating refined embeddings directly into a compatible diffusion-model conditioning pipeline.
- Training against image-grounded spatial objectives rather than only embedding-based losses.
- Evaluating generated images with established spatial-relation benchmarks.
- Comparing results against an unmodified baseline using identical prompts and sampling configurations.
- Improving the synthetic caption generation process with more reliable object and relation annotations.
- Adding controlled experiments, ablation studies, multiple random seeds, and statistical significance testing.
- Documenting package versions, GPU requirements, dataset provenance, and reproducibility settings.

## Author

**Laiba Imran**

GitHub: [@LaibaImran1500](https://github.com/LaibaImran1500)

## Repository

[SGRA2500-ML on GitHub](https://github.com/LaibaImran1500/SGRA2500-ML)

## License

Refer to the repository for license information. Ensure that the licenses and usage terms of the underlying datasets and pretrained models are respected.
