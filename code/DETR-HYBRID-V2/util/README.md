# Report Generation & Explainable AI Scripts

This directory contains utility scripts for generating comprehensive reports and explainable AI analysis from DETR-HYBRID-V2 model training outputs.

## 📊 Scripts Overview

### 1. **report_generator.py** - Metrics Report Generation

Generates Excel workbooks with comprehensive training, validation, and test metrics formatted for thesis writing and model comparison.

#### Features
- **Multi-sheet Excel output** with organized metric categories
- **Detection metrics**: AP/AR across IoU thresholds, per-class (fire/smoke)
- **Efficiency metrics**: Inference time, GFLOPs, token pruning ratios (HYBRID-V2 only)
- **Loss components**: CE, BBox, GIoU losses
- **Training summary**: Epochs, parameters, final metrics
- **Epoch-wise curves**: Training loss, learning rate progression

#### Usage

**DETR Model:**
```bash
cd /root/fire-and-smoke-detection-thesis
python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "Pretrained-2"
```

**DETR-HYBRID-V2 Model:**
```bash
cd /root/fire-and-smoke-detection-thesis
python code/DETR-HYBRID-V2/util/report_generator.py \
    --output_dir code/DETR-HYBRID-V2/outputs \
    --experiment_name "2-withwarmingepoch"
```

#### Arguments
- `--output_dir` (required): Path to model output directory
- `--experiment_name` (required): Experiment folder name (e.g., "Pretrained-2", "2-withwarmingepoch")
- `--report_path` (optional): Custom output path for Excel file (auto-generated if not specified)

#### Output Files

Generated in the experiment directory:
- `report_YYYYMMDD_HHMMSS.xlsx` - Excel workbook with sheets:
  - **Summary**: Single row with all key metrics (optimal for thesis tables)
  - **Detection Metrics**: Full COCO AP/AR values
  - **Per-Class Metrics**: Fire/Smoke specific performance
  - **Efficiency & Pruning**: Performance metrics and pruning ratios (HYBRID-V2)
  - **Training Curves**: Epoch-wise metrics for reference

#### Example Output (Summary Sheet)

| Model | COCO AP (IoU=0.50:0.95) | COCO AP (IoU=0.50) | AP_fire | Recall_fire | Forward Time (ms) | Pixel Keep Ratio | GFLOPs (After) |
|-------|------------------------|-------------------|---------|-------------|------------------|-----------------|----------------|
| Pretrained-2 | 0.4521 | 0.6892 | 0.521 | 0.676 | 45.2 | 0.80 | 125.3 |

---

### 2. **explainable_ai.py** - DETR-HYBRID-V2 Interpretability Analysis

Comprehensive analysis of the token pruning mechanism, attention patterns, and feature importance to explain model decisions.

#### Features

1. **Pruning Decision Visualization**
   - Before/after pruning comparisons
   - Visual representation of which regions are kept vs. pruned
   - Correctness verification: ensures fire/smoke regions preserved

2. **Attention Pattern Analysis**
   - Decoder cross-attention heatmaps showing focus regions
   - Query-to-image attention distribution
   - Identification of attended vs. non-attended tokens

3. **Feature Importance Analysis**
   - Gradient-based importance scoring
   - Per-layer importance ranking
   - Backbone and transformer feature contribution analysis

4. **Decoder Efficiency Analysis**
   - Token utilization statistics
   - Comparison of pruned vs. non-pruned token importance
   - Accuracy/efficiency trade-off quantification

#### Usage

```bash
cd /root/fire-and-smoke-detection-thesis

python code/DETR-HYBRID-V2/util/explainable_ai.py \
    --model_path code/DETR-HYBRID-V2/outputs/2-withwarmingepoch/checkpoint.pth \
    --output_dir code/DETR-HYBRID-V2/outputs/2-withwarmingepoch \
    --num_samples 10 \
    --device cuda
```

#### Arguments
- `--model_path` (required): Path to trained checkpoint.pth file
- `--output_dir` (required): Directory where analysis outputs will be saved
- `--num_samples` (optional, default=10): Number of test samples to analyze
- `--device` (optional, default='cuda'): Device to run on ('cuda' or 'cpu')

#### Output Structure

```
xai_analysis/
├── pruning_analysis/
│   ├── before_after_000.png        # Visual before/after pruning
│   ├── before_after_001.png
│   ├── pruning_correctness.csv     # Pruning statistics per sample
│   └── pruning_statistics.json     # Aggregate pruning metrics
├── attention_analysis/
│   ├── attention_heatmap_000.png   # Decoder attention patterns
│   ├── attention_heatmap_001.png
│   └── attention_metrics.csv       # Quantitative attention analysis
├── feature_importance/
│   ├── feature_importance_by_layer.csv
│   └── feature_importance_chart.png
├── decoder_analysis/
│   ├── token_utilization.csv       # Token usage statistics
│   └── decoder_efficiency.json
└── summary_report.json             # Overall findings
```

#### Key Metrics Generated

**Pruning Analysis:**
- `num_predictions`: Number of objects detected
- `attention_concentration`: How focused decoder attention is (0-1, higher = more focused)
- `attention_entropy`: Information content of attention (higher = more spread)

**Feature Importance:**
- `mean_importance`: Average layer importance score
- `per_layer`: Importance for each transformer layer

**Token Utilization:**
- `attended_tokens`: Number of tokens actually attended to
- `total_tokens`: Total tokens after pruning
- `utilization_ratio`: Proportion of tokens used by decoder

---

## 📈 Quick Start Examples

### Generate comparison report for thesis

```bash
# Generate DETR report
python code/DETR/util/report_generator.py \
    --output_dir code/DETR/outputs \
    --experiment_name "Pretrained-2"

# Generate DETR-HYBRID-V2 report
python code/DETR-HYBRID-V2/util/report_generator.py \
    --output_dir code/DETR-HYBRID-V2/outputs \
    --experiment_name "2-withwarmingepoch"
```

Then open both Excel files and copy the Summary sheet rows into your thesis tables.

### Generate XAI analysis for model explanation chapter

```bash
python code/DETR-HYBRID-V2/util/explainable_ai.py \
    --model_path code/DETR-HYBRID-V2/outputs/2-withwarmingepoch/checkpoint.pth \
    --output_dir code/DETR-HYBRID-V2/outputs/2-withwarmingepoch \
    --num_samples 20 \
    --device cuda
```

Review the PNG visualizations in `xai_analysis/pruning_analysis/` and `xai_analysis/attention_analysis/` for your thesis figures.

---

## 🔧 Requirements

### Dependencies
```bash
pip install torch torchvision pandas openpyxl matplotlib opencv-python numpy
```

### For Report Generation
- pandas
- openpyxl

### For XAI Analysis
- torch
- torchvision
- matplotlib
- opencv-python (cv2)
- numpy

---

## 📝 Output Format Notes

### Excel Reports

All numeric values are formatted to 4 decimal places for precision. Column widths auto-adjust based on content. Use "Summary" sheet for direct copy-paste into thesis tables.

**Timestamp format**: Reports are named `report_YYYYMMDD_HHMMSS.xlsx` to prevent overwrites.

### XAI Visualizations

- **PNG images**: 100 DPI, suitable for thesis paper inclusion
- **CSV files**: Comma-separated, compatible with Excel/Google Sheets
- **JSON files**: Structured metadata for detailed analysis

---

## 💡 Tips for Thesis Writing

### Using Report Metrics

1. Open the Excel report in your preferred spreadsheet application
2. Navigate to the "Summary" sheet
3. Copy the entire row (all columns) for your model
4. Paste into your thesis table or create a comparison table with multiple model rows

### Using XAI Visualizations

1. **Before/After Pruning**: Shows the effectiveness of token reduction strategy
   - Verify that background regions are pruned (high token removal)
   - Verify that fire/smoke regions are preserved (low pruning in important areas)

2. **Attention Heatmaps**: Demonstrates what the model learns to focus on
   - Include 2-3 examples showing clear attention on fire/smoke objects
   - Use for explaining model interpretability

3. **Feature Importance Charts**: Shows layer contribution to predictions
   - Explains which transformer layers are most critical

### Metrics Interpretation

- **mAP50**: Mean Average Precision at IoU=0.50 - primary metric for detection
- **AP_fire / AP_smoke**: Class-specific performance - important for thesis
- **Pixel Keep Ratio**: Percentage of pixels kept after pruning (0.8 = 80% retained)
- **GFLOPs After**: Computational cost after optimization (lower is better)
- **Forward Time**: Inference latency in milliseconds

---

## ✅ Verification Checklist

- [ ] Reports generated without errors
- [ ] Excel files open and display all metrics
- [ ] XAI visualizations show clear before/after pruning differences
- [ ] Attention heatmaps highlight fire/smoke regions
- [ ] Feature importance charts show reasonable layer distributions
- [ ] CSV files are readable and contain expected columns
- [ ] Summary report JSON has valid structure

---

## 📧 Notes

- Reports read from existing `training_log.json` and `test_log.json` files
- No model retraining needed - purely analysis of completed experiments
- Analysis is non-invasive and doesn't modify checkpoint files
- All outputs are self-contained in subdirectories

