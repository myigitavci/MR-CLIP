# MR-CLIP: Efficient Metadata-Guided Learning of Robust MRI Contrast Representations

<div align="center">
  <img src="docs/mr-clip-overview.png" alt="MR-CLIP Overview" width="800"/>
</div>

## Abstract

<div align="justify">

**Challenge:** The interpretation and analysis of Magnetic Resonance Imaging scans in clinical AI systems rely on accurate understanding of image contrast. While contrast is determined by acquisition parameters stored in DICOM metadata, real-world clinical datasets often suffer from noisy, incomplete, or inconsistent metadata. Broad labels like 'T1 weighted' or 'T2-weighted' are commonly used, but offer only coarse and insufficient descriptions.

**Problem:** In many real-world clinical datasets, such labels are missing altogether, leaving raw acquisition parameters, such as echo time and repetition time, as the only available contrast indicators. These parameters directly govern image appearance and are critical for accurate interpretation, reliable retrieval, and integration into clinical workflows.

**Solution:** To address these challenges, we propose MR-CLIP, a multimodal contrastive learning framework that aligns MR images with their DICOM metadata to learn contrast-aware representations, without manual labels. Trained on diverse clinical data spanning various scanners and protocols, MR-CLIP captures contrast variation across acquisitions and within scans, enabling anatomy-independent representation learning.

</div>

## Repository Structure

```
MR-CLIP/
├── preprocessing.py          # Main preprocessing script
├── preprocessing.ipynb       # Jupyter notebook version of preprocessing
├── bin_intervals_et_20_rt_20.json  # Binning intervals for Echo Time and Repetition Time
├── requirements.txt          # Python dependencies
└── src/                     # Source code for training and evaluation
```

## Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n mr-clip python=3.8 -y
conda activate mr-clip

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

The preprocessing pipeline consists of four main steps:

1. **NIfTI to PNG Conversion**
   - Converts NIfTI files to PNG images
   - Determines scanning plane (axial, coronal, sagittal)
   - Normalizes pixel values to 0-255
   - Saves only nonzero slices

2. **CSV Creation**
   - Creates CSV files with image paths and metadata
   - Extracts and simplifies DICOM metadata
   - Organizes data in batches for efficient processing

3. **Data Labeling**
   - Creates unique labels for parameter combinations
   - Bins numerical values (Echo Time, Repetition Time)
   - Tracks label distributions

4. **Data Splitting**
   - Merges all labeled data
   - Shuffles while keeping slices together
   - Splits into train/val/test sets

To run the preprocessing:
```bash
# Open and run the preprocessing notebook
jupyter notebook preprocessing.ipynb
```

### 3. Testing on Your Data

1. **Download Pre-trained Weights**
   - Download the pre-trained weights from [LINK_TO_BE_ADDED]
   - Place the weights file in the `logs/mr_clip/checkpoints` directory

2. **Prepare Your Test Data**
   - Ensure your test data is in CSV format with columns:
     - `filepath`: Path to the PNG image
     - `text`: Text description/metadata
     - `label`: Numerical label (if available)

3. **Run Testing**
```bash
python -m open_clip_train.main \
    --report-to tensorboard \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --val-data=/path/to/your/test_data.csv \
    --batch-size=1000 \
    --workers=8 \
    --logs=/path/to/logs \
    --device=cuda \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --name=mr_clip \
    --resume=latest \
    --distance \
    --test \
    --tracepreds
```

Key parameters to adjust:
- `--val-data`: Path to your test data CSV
- `--batch-size`: Adjust based on your GPU memory
- `--workers`: Number of data loading workers
- `--logs`: Directory for saving logs
- `--name`: Name of the experiment under logs folder. Place the weights under this folder.

With the  `--tracepreds ` option, you can check the results of retrieved predictions case by case which are saved under  `mr_clip/checkpoints `.

## Weights

Will be made public upon acceptance.

## Citation

If you use this code in your research, please cite:
```bibtex
[Your citation information here]
```


## Contact

For questions and issues, please open an issue in this repository.

## Acknowledgement

Our code repository is mainly built on [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors for releasing their code.
