

# This repository is moved to https://github.com/myigitavci/MaRaI not os maintained


## 🗂️ Repository Structure

```
MR-CLIP/
├── preprocessing.py          # Main preprocessing script
├── preprocessing.ipynb       # Jupyter notebook version of preprocessing
├── bin_intervals_et_20_rt_20.json  # Binning intervals for Echo Time and Repetition Time
├── requirements.txt          # Python dependencies
└── src/                     # Source code for training and evaluation
```
---

## 🚀 Quick Start

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
   - Download the pre-trained weights from [[20x20 Weights]](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing)
   - Place the weights file in the `logs/mr_clip/checkpoints` directory

2. **Prepare Your Test Data**
   - Ensure your test data is in CSV format with columns:
     - `filepath`: Path to the PNG image
     - `text`: Text description/metadata
     - `label`: Numerical label (if available)

3. **Run Testing**
```bash
cd src
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

[20x20 Weights](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing)

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{avci2025mrclipefficientmetadataguidedlearning,
      title={MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations}, 
      author={Mehmet Yigit Avci and Pedro Borges and Paul Wright and Mehmet Yigitsoy and Sebastien Ourselin and Jorge Cardoso},
      year={2025},
      eprint={2507.00043},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.00043}, 
}
```


## Contact

For questions and issues, please open an issue in this repository.

## Acknowledgement

Our code repository is mainly built on [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors for releasing their code.
