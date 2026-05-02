"""
Report Generation Script for DETR-HYBRID-V2 Model
==================================================
Generates comprehensive Excel reports including efficiency metrics and pruning statistics
for DETR-HYBRID-V2 model results.

Usage:
    python report_generator.py --output_dir /path/to/outputs --experiment_name "2-withwarmingepoch"
    
Outputs:
    - report_{timestamp}.xlsx with multiple sheets:
        - Summary: Single row with all key metrics
        - Detection Metrics: Full COCO AP/AR across IoU levels
        - Per-Class Metrics: Fire/Smoke specific metrics
        - Efficiency: Pruning ratios, inference time, GFLOPs, token reduction
        - Training Curves: Epoch-wise metrics for reference
"""

import json
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ReportGeneratorHybrid:
    """Generate Excel reports from DETR-HYBRID-V2 training outputs."""
    
    # COCO AP/AR metric indices
    COCO_METRIC_NAMES = [
        'AP (IoU=0.50:0.95)',
        'AP (IoU=0.50)',
        'AP (IoU=0.75)',
        'AP small',
        'AP medium',
        'AP large',
        'AR (IoU=0.50:0.95)',
        'AR (IoU=0.50)',
        'AR (IoU=0.75)',
        'AR small',
        'AR medium',
        'AR large'
    ]
    
    def __init__(self, output_dir: str, experiment_name: str):
        """
        Initialize report generator for DETR-HYBRID-V2.
        
        Args:
            output_dir: Path to model output directory
            experiment_name: Name of the experiment (e.g., "2-withwarmingepoch")
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.exp_path = self.output_dir / experiment_name
        
        if not self.exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.exp_path}")
        
        self.training_log = self._load_json('training_log.json')
        self.test_log = self._load_json('test_log.json')
        self.info = self._load_info()
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from experiment directory."""
        filepath = self.exp_path / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found at {filepath}")
            return {} if filename == 'test_log.json' else []
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_info(self) -> Dict:
        """Load training info from info.txt."""
        filepath = self.exp_path / 'info.txt'
        if not filepath.exists():
            return {}
        
        info = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, val = line.split(':', 1)
                    info[key.strip()] = val.strip()
        return info
    
    def get_final_metrics(self) -> Dict:
        """
        Extract final test metrics from logs.
        
        Returns:
            Dictionary with all available metrics
        """
        if isinstance(self.training_log, list) and len(self.training_log) > 0:
            # Get last epoch metrics (typically contains test metrics)
            last_epoch = self.training_log[-1]
        else:
            last_epoch = {}
        
        # Merge with test_log if available
        final_metrics = {**last_epoch, **self.test_log}
        
        return final_metrics
    
    def extract_coco_metrics(self, final_metrics: Dict) -> Dict:
        """Extract COCO AP/AR metrics."""
        coco_dict = {}
        coco_values = final_metrics.get('test_coco_eval_bbox', [])
        
        if isinstance(coco_values, list) and len(coco_values) >= 12:
            for idx, name in enumerate(self.COCO_METRIC_NAMES):
                coco_dict[name] = round(float(coco_values[idx]), 4)
        
        return coco_dict
    
    def extract_per_class_metrics(self, final_metrics: Dict) -> Dict:
        """Extract per-class detection metrics (Fire/Smoke)."""
        class_dict = {}
        
        metrics_map = {
            'AP_fire': 'test_AP_fire',
            'Recall_fire': 'test_Recall_fire',
            'AP_smoke': 'test_AP_smoke',
            'Recall_smoke': 'test_Recall_smoke'
        }
        
        for display_name, metric_key in metrics_map.items():
            if metric_key in final_metrics:
                class_dict[display_name] = round(float(final_metrics[metric_key]), 4)
        
        return class_dict
    
    def extract_loss_metrics(self, final_metrics: Dict) -> Dict:
        """Extract loss components from final metrics."""
        loss_dict = {}
        
        loss_keys = [
            ('test_loss', 'Total Loss'),
            ('test_loss_ce', 'CE Loss'),
            ('test_loss_bbox', 'BBox Loss'),
            ('test_loss_giou', 'GIoU Loss'),
        ]
        
        for key, display_name in loss_keys:
            if key in final_metrics:
                loss_dict[display_name] = round(float(final_metrics[key]), 4)
        
        return loss_dict
    
    def extract_efficiency_metrics(self, final_metrics: Dict) -> Dict:
        """Extract efficiency metrics specific to DETR-HYBRID-V2."""
        eff_dict = {}
        
        # Standard efficiency metrics
        eff_keys = [
            ('test_eff_forward_ms', 'Forward Time (ms)'),
            ('test_eff_imgs_per_s', 'Imgs/sec'),
            ('test_eff_iter_ms', 'Iteration Time (ms)'),
        ]
        
        for key, display_name in eff_keys:
            if key in final_metrics:
                val = float(final_metrics[key])
                eff_dict[display_name] = round(val, 2)
        
        # Pruning-specific metrics
        pruning_keys = [
            ('test_eff_pixel_prune_enabled', 'Pruning Enabled'),
            ('test_eff_pixel_keep_ratio_actual', 'Pixel Keep Ratio'),
        ]
        
        for key, display_name in pruning_keys:
            if key in final_metrics:
                val = final_metrics[key]
                if isinstance(val, bool):
                    eff_dict[display_name] = str(val)
                else:
                    eff_dict[display_name] = round(float(val), 4)
        
        # Token reduction metrics
        token_keys = [
            ('test_eff_encoder_seq_len_before_prune', 'Encoder Tokens (Before)'),
            ('test_eff_encoder_seq_len_after_prune', 'Encoder Tokens (After)'),
        ]
        
        for key, display_name in token_keys:
            if key in final_metrics:
                val = final_metrics[key]
                eff_dict[display_name] = int(val)
        
        # GFLOPs metrics
        gflops_keys = [
            ('test_eff_gflops_before_prune', 'GFLOPs (Before)'),
            ('test_eff_gflops_after_prune', 'GFLOPs (After)'),
        ]
        
        for key, display_name in gflops_keys:
            if key in final_metrics:
                val = final_metrics[key]
                eff_dict[display_name] = round(float(val), 2)
        
        return eff_dict
    
    def extract_training_summary(self, final_metrics: Dict) -> Dict:
        """Extract training summary statistics."""
        summary = {}
        
        # Get epoch info
        if isinstance(self.training_log, list) and len(self.training_log) > 0:
            summary['Epochs Trained'] = int(self.training_log[-1].get('epoch', len(self.training_log)))
            summary['Final Train Loss'] = round(float(self.training_log[-1].get('train_loss', 0)), 4)
            summary['Final Learning Rate'] = self.training_log[-1].get('train_lr', 0)
        
        # Model info
        if isinstance(self.training_log, list) and len(self.training_log) > 0:
            if 'n_parameters' in self.training_log[-1]:
                n_params = int(self.training_log[-1]['n_parameters'])
                summary['Model Parameters'] = f"{n_params:,}"
        
        # Dataset info
        summary['Model Variant'] = self.experiment_name
        
        return summary
    
    def get_epoch_wise_metrics(self) -> pd.DataFrame:
        """Extract epoch-wise metrics for training curves, including efficiency."""
        if not isinstance(self.training_log, list):
            return pd.DataFrame()
        
        epochs_data = []
        for epoch_data in self.training_log:
            epoch_row = {
                'Epoch': epoch_data.get('epoch', ''),
                'Train Loss': epoch_data.get('train_loss', None),
                'Train LR': epoch_data.get('train_lr', None),
                'Test Loss': epoch_data.get('test_loss', None),
                'Test mAP50': epoch_data.get('test_mAP50', None),
                'Test Class Error': epoch_data.get('test_class_error', None),
            }
            
            # Add efficiency metrics if available
            if 'test_eff_pixel_keep_ratio_actual' in epoch_data:
                epoch_row['Pixel Keep Ratio'] = round(float(epoch_data['test_eff_pixel_keep_ratio_actual']), 4)
            
            if 'test_eff_forward_ms' in epoch_data:
                epoch_row['Forward Time (ms)'] = round(float(epoch_data['test_eff_forward_ms']), 2)
            
            epochs_data.append(epoch_row)
        
        return pd.DataFrame(epochs_data)
    
    def create_summary_sheet(self, final_metrics: Dict) -> pd.DataFrame:
        """Create summary sheet with all key metrics in one row."""
        summary_dict = {}
        
        # Add experiment name
        summary_dict['Model'] = self.experiment_name
        
        # Detection metrics
        coco_metrics = self.extract_coco_metrics(final_metrics)
        summary_dict.update({f"COCO {k}": v for k, v in coco_metrics.items()})
        
        # Per-class metrics
        class_metrics = self.extract_per_class_metrics(final_metrics)
        summary_dict.update(class_metrics)
        
        # Loss metrics
        loss_metrics = self.extract_loss_metrics(final_metrics)
        summary_dict.update(loss_metrics)
        
        # Efficiency metrics (including pruning)
        eff_metrics = self.extract_efficiency_metrics(final_metrics)
        summary_dict.update(eff_metrics)
        
        # Training summary
        train_summary = self.extract_training_summary(final_metrics)
        summary_dict.update(train_summary)
        
        return pd.DataFrame([summary_dict])
    
    def create_detection_metrics_sheet(self, final_metrics: Dict) -> pd.DataFrame:
        """Create detailed detection metrics sheet."""
        coco_metrics = self.extract_coco_metrics(final_metrics)
        
        # Convert to two columns: Metric and Value
        data = [
            {'Metric': k, 'Value': v}
            for k, v in coco_metrics.items()
        ]
        
        return pd.DataFrame(data)
    
    def create_per_class_sheet(self, final_metrics: Dict) -> pd.DataFrame:
        """Create per-class metrics sheet."""
        class_metrics = self.extract_per_class_metrics(final_metrics)
        
        data = [
            {'Class Metric': k, 'Value': v}
            for k, v in class_metrics.items()
        ]
        
        return pd.DataFrame(data)
    
    def create_efficiency_sheet(self, final_metrics: Dict) -> pd.DataFrame:
        """Create comprehensive efficiency and pruning metrics sheet."""
        eff_metrics = self.extract_efficiency_metrics(final_metrics)
        loss_metrics = self.extract_loss_metrics(final_metrics)
        
        data = []
        
        # Efficiency section
        data.append({'Category': 'Performance', 'Metric': '', 'Value': ''})
        for k, v in eff_metrics.items():
            data.append({'Category': 'Performance', 'Metric': k, 'Value': v})
        
        # Loss section
        data.append({'Category': 'Loss', 'Metric': '', 'Value': ''})
        for k, v in loss_metrics.items():
            data.append({'Category': 'Loss', 'Metric': k, 'Value': v})
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Excel report with efficiency metrics.
        
        Args:
            output_path: Path to save Excel file. If None, auto-generates filename.
            
        Returns:
            Path to generated Excel file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.exp_path / f'report_{timestamp}.xlsx'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_metrics = self.get_final_metrics()
        
        # Create sheets
        summary_df = self.create_summary_sheet(final_metrics)
        detection_df = self.create_detection_metrics_sheet(final_metrics)
        per_class_df = self.create_per_class_sheet(final_metrics)
        efficiency_df = self.create_efficiency_sheet(final_metrics)
        epoch_df = self.get_epoch_wise_metrics()
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            detection_df.to_excel(writer, sheet_name='Detection Metrics', index=False)
            per_class_df.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            
            if not efficiency_df.empty:
                efficiency_df.to_excel(writer, sheet_name='Efficiency & Pruning', index=False)
            
            if not epoch_df.empty:
                epoch_df.to_excel(writer, sheet_name='Training Curves', index=False)
            
            # Auto-adjust column widths
            for sheet in writer.sheets.values():
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    sheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✓ Report generated: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive Excel reports for DETR-HYBRID-V2 model results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to model output directory'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Experiment name (e.g., "2-withwarmingepoch", "1")'
    )
    parser.add_argument(
        '--report_path',
        type=str,
        default=None,
        help='Path to save report (auto-generated if not specified)'
    )
    
    args = parser.parse_args()
    
    try:
        generator = ReportGeneratorHybrid(args.output_dir, args.experiment_name)
        report_path = generator.generate_report(args.report_path)
        print(f"Report successfully created at: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        raise


if __name__ == '__main__':
    main()
