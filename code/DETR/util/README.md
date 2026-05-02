# Report Generation for DETR

This directory contains utility scripts for generating comprehensive reports from DETR model training outputs.

## 📊 Script: report_generator.py

Generates Excel workbooks with comprehensive training, validation, and test metrics formatted for thesis writing and model comparison.

### Features
- **Multi-sheet Excel output** with organized metric categories
- **Detection metrics**: AP/AR across IoU thresholds, per-class (fire/smoke)
- **Loss components**: CE, BBox, GIoU losses
- **Training summary**: Epochs, parameters, final metrics
- **Epoch-wise curves**: Training loss, learning rate progression

### Usage

```bash
cd /root/fire-and-smoke-detection-thesis

python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "Pretrained-2"
```

### Arguments
- `--output_dir` (required): Path to model output directory
- `--experiment_name` (required): Experiment folder name (e.g., "Pretrained-2", "from_scratch", "Pretrained")
- `--report_path` (optional): Custom output path for Excel file (auto-generated if not specified)

### Output Files

Generated in the experiment directory:
- `report_YYYYMMDD_HHMMSS.xlsx` - Excel workbook with sheets:
  - **Summary**: Single row with all key metrics (optimal for thesis tables)
  - **Detection Metrics**: Full COCO AP/AR values
  - **Per-Class Metrics**: Fire/Smoke specific performance
  - **Efficiency**: Performance metrics
  - **Training Curves**: Epoch-wise metrics for reference

### Example Output (Summary Sheet)

| Model | COCO AP (IoU=0.50:0.95) | COCO AP (IoU=0.50) | AP_fire | Recall_fire | Epochs Trained |
|-------|------------------------|-------------------|---------|-------------|----------------|
| Pretrained-2 | 0.4521 | 0.6892 | 0.521 | 0.676 | 50 |

---

## 🔧 Requirements

```bash
pip install pandas openpyxl
```

---

## 💡 Tips for Thesis Writing

### Using Report Metrics

1. Open the Excel report in your preferred spreadsheet application
2. Navigate to the "Summary" sheet
3. Copy the entire row (all columns) for your model
4. Paste into your thesis table or create a comparison table with multiple model rows

### Available Experiments

Generate reports for all DETR variants:

```bash
# Pretrained with ResNet50
python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "Pretrained"

# Pretrained variant 2
python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "Pretrained-2"

# From scratch (no pretraining)
python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "from_scratch"
```

---

## 📝 Metrics Included

### COCO Detection Metrics
- AP @ IoU=0.50:0.95 (main metric)
- AP @ IoU=0.50
- AP @ IoU=0.75
- AP for small/medium/large objects
- AR (Average Recall) variants

### Per-Class Metrics
- AP_fire: Fire detection average precision
- Recall_fire: Fire detection recall
- AP_smoke: Smoke detection average precision
- Recall_smoke: Smoke detection recall

### Loss Components
- Total Loss
- CE Loss (classification)
- BBox Loss (bounding box regression)
- GIoU Loss (generalized IoU)

### Training Summary
- Epochs Trained
- Final Train Loss
- Final Learning Rate
- Model Parameters

---

## ✅ Verification Checklist

- [ ] Report generated without errors
- [ ] Excel file opens and displays all metrics
- [ ] All sheets are populated with data
- [ ] Numbers match the training logs
- [ ] Summary sheet has exactly one row
