"""
Clean training logger for better readability during training.
"""
import json
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """Logger for clean epoch summaries and metric tracking."""
    
    def __init__(self, output_dir, model_name="DETR"):
        self.output_dir = Path(output_dir)
        self.logs_dir = Path('outputs/logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.log_file = self.logs_dir / 'log.txt'
        self.metrics_file = self.logs_dir / 'metrics_summary.txt'
        self.best_metrics = {'val_loss': float('inf'), 'val_AP': 0.0, 'best_epoch': -1}
        
        # Initialize metrics summary file
        with open(self.metrics_file, 'w') as f:
            f.write(f"Training Log for {model_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
    
    def log_epoch(self, epoch, train_stats, val_stats, epoch_time):
        """
        Log epoch with clean formatting showing key metrics only.
        
        Args:
            epoch: Current epoch number
            train_stats: Dict with training metrics
            val_stats: Dict with validation metrics (can be empty)
            epoch_time: Time taken for epoch in seconds
        """
        # Save detailed JSON log (unchanged)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'epoch_time': epoch_time
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_stats) + "\n")
        
        # Print clean epoch summary
        print("\n" + "="*100)
        print(f"EPOCH {epoch} SUMMARY".center(100))
        print("="*100)
        
        # Training metrics
        print(f"\n{'TRAINING:':<20} Loss: {train_stats.get('loss', 0):.4f} | " +
              f"Class Error: {train_stats.get('class_error', 0):.2f}% | " +
              f"LR: {train_stats.get('lr', 0):.6f}")
        
        if train_stats.get('loss_ce') is not None:
            print(f"{'':20} CE: {train_stats.get('loss_ce', 0):.4f} | " +
                  f"BBox: {train_stats.get('loss_bbox', 0):.4f} | " +
                  f"GIoU: {train_stats.get('loss_giou', 0):.4f}")
        
        # Validation metrics
        if val_stats:
            val_loss = val_stats.get('loss', 0)
            val_class_err = val_stats.get('class_error', 0)
            
            print(f"\n{'VALIDATION:':<20} Loss: {val_loss:.4f} | " +
                  f"Class Error: {val_class_err:.2f}%")
            
            # Detection metrics (AP, Recall) - only if available
            has_coco_metrics = 'AP' in val_stats or 'mAP50' in val_stats
            
            if has_coco_metrics:
                print(f"{'':20} mAP: {val_stats.get('AP', 0):.4f} | " +
                      f"mAP50: {val_stats.get('mAP50', 0):.4f} | " +
                      f"Recall: {val_stats.get('Recall', 0):.4f}")
                
                # Per-class metrics - only show if available
                if 'AP_fire' in val_stats or 'AP_smoke' in val_stats:
                    print(f"\n{'PER-CLASS METRICS:':<20}")
                    if 'AP_fire' in val_stats:
                        print(f"  {'Fire:':<18} AP: {val_stats.get('AP_fire', 0):.4f} | " +
                              f"AP50: {val_stats.get('AP50_fire', 0):.4f} | " +
                              f"Recall: {val_stats.get('Recall_fire', 0):.4f}")
                    if 'AP_smoke' in val_stats:
                        print(f"  {'Smoke:':<18} AP: {val_stats.get('AP_smoke', 0):.4f} | " +
                              f"AP50: {val_stats.get('AP50_smoke', 0):.4f} | " +
                              f"Recall: {val_stats.get('Recall_smoke', 0):.4f}")
            else:
                # No COCO metrics available
                print(f"{'':20} ⚠ COCO metrics unavailable (install pycocotools)")
        
        # Time and best metrics
        print(f"\n{'TIME:':<20} {epoch_time:.1f}s")
        
        # Update best metrics
        improvement = []
        if val_stats:
            if val_stats.get('loss', float('inf')) < self.best_metrics['val_loss']:
                self.best_metrics['val_loss'] = val_stats['loss']
                self.best_metrics['best_epoch'] = epoch
                improvement.append(f"Best Val Loss: {self.best_metrics['val_loss']:.4f} ✓")
            
            if val_stats.get('AP', 0) > self.best_metrics['val_AP']:
                self.best_metrics['val_AP'] = val_stats['AP']
                improvement.append(f"Best mAP: {self.best_metrics['val_AP']:.4f} ✓")
        
        if improvement:
            print(f"\n{'IMPROVEMENTS:':<20} {' | '.join(improvement)}")
        
        print("="*100 + "\n")
        
        # Save to metrics summary
        with open(self.metrics_file, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train - Loss: {train_stats.get('loss', 0):.4f}\n")
            
            if val_stats:
                # Overall average
                f.write(f"  Val (Avg) - Loss: {val_stats.get('loss', 0):.4f}")
                if 'AP' in val_stats:
                    f.write(f" | mAP: {val_stats.get('AP', 0):.4f} | Recall: {val_stats.get('Recall', 0):.4f}")
                
                # Per-class averages
                if 'AP_fire' in val_stats:
                     f.write(f"\n    Fire (Avg) - AP: {val_stats.get('AP_fire', 0):.4f} | Recall: {val_stats.get('Recall_fire', 0):.4f}")
                if 'AP_smoke' in val_stats:
                     f.write(f"\n    Smoke (Avg) - AP: {val_stats.get('AP_smoke', 0):.4f} | Recall: {val_stats.get('Recall_smoke', 0):.4f}")
                
                f.write("\n")
                
                # Per-category breakdown
                categories = ['CV', 'UAV', 'RS']
                for cat in categories:
                    if f'{cat}_loss' in val_stats:
                        f.write(f"  Val ({cat}) - Loss: {val_stats.get(f'{cat}_loss', 0):.4f}")
                        if f'{cat}_AP' in val_stats:
                            f.write(f" | mAP: {val_stats.get(f'{cat}_AP', 0):.4f} | Recall: {val_stats.get(f'{cat}_Recall', 0):.4f}")
                        f.write("\n")
                        
                        # Per-class within category
                        if f'{cat}_AP_fire' in val_stats:
                             f.write(f"    Fire  - AP: {val_stats.get(f'{cat}_AP_fire', 0):.4f} | Recall: {val_stats.get(f'{cat}_Recall_fire', 0):.4f}\n")
                        if f'{cat}_AP_smoke' in val_stats:
                             f.write(f"    Smoke - AP: {val_stats.get(f'{cat}_AP_smoke', 0):.4f} | Recall: {val_stats.get(f'{cat}_Recall_smoke', 0):.4f}\n")

            f.write(f"  Time: {epoch_time:.1f}s\n\n")
    
    def log_final_results(self, total_time, test_stats=None):
        """Log final training results."""
        print("\n" + "="*100)
        print("TRAINING COMPLETE".center(100))
        print("="*100)
        print(f"\nTotal Time: {total_time}")
        print(f"Best Validation Loss: {self.best_metrics['val_loss']:.4f} (Epoch {self.best_metrics['best_epoch']})")
        print(f"Best Validation mAP: {self.best_metrics['val_AP']:.4f}")
        
        if test_stats:
            print(f"\n{'TEST SET RESULTS:':<20}")
            print(f"{'':20} Loss: {test_stats.get('loss', 0):.4f}")
            if 'AP' in test_stats:
                print(f"{'':20} mAP: {test_stats.get('AP', 0):.4f} | " +
                      f"mAP50: {test_stats.get('mAP50', 0):.4f} | " +
                      f"Recall: {test_stats.get('Recall', 0):.4f}")
        
        print("="*100 + "\n")
        
        # Save final summary
        with open(self.metrics_file, 'a') as f:
            f.write("\n" + "="*100 + "\n")
            f.write(f"FINAL RESULTS\n")
            f.write("="*100 + "\n")
            f.write(f"Total Time: {total_time}\n")
            f.write(f"Best Val Loss: {self.best_metrics['val_loss']:.4f} (Epoch {self.best_metrics['best_epoch']})\n")
            f.write(f"Best Val mAP: {self.best_metrics['val_AP']:.4f}\n")
            if test_stats:
                f.write(f"\nTest Results:\n")
                f.write(f"  Loss: {test_stats.get('loss', 0):.4f}\n")
                if 'AP' in test_stats:
                    f.write(f"  mAP: {test_stats.get('AP', 0):.4f} | mAP50: {test_stats.get('mAP50', 0):.4f} | Recall: {test_stats.get('Recall', 0):.4f}\n")
                    
                    # Per-class averages
                    if 'AP_fire' in test_stats:
                         f.write(f"  Fire - AP: {test_stats.get('AP_fire', 0):.4f} | Recall: {test_stats.get('Recall_fire', 0):.4f}\n")
                    if 'AP_smoke' in test_stats:
                         f.write(f"  Smoke - AP: {test_stats.get('AP_smoke', 0):.4f} | Recall: {test_stats.get('Recall_smoke', 0):.4f}\n")
                    
                    # Per-category breakdown
                    categories = ['CV', 'UAV', 'RS']
                    for cat in categories:
                        if f'{cat}_loss' in test_stats:
                            f.write(f"\n  {cat} Set:\n")
                            f.write(f"    Loss: {test_stats.get(f'{cat}_loss', 0):.4f}\n")
                            if f'{cat}_AP' in test_stats:
                                f.write(f"    mAP: {test_stats.get(f'{cat}_AP', 0):.4f} | Recall: {test_stats.get(f'{cat}_Recall', 0):.4f}\n")
                            if f'{cat}_AP_fire' in test_stats:
                                f.write(f"    Fire - AP: {test_stats.get(f'{cat}_AP_fire', 0):.4f} | Recall: {test_stats.get(f'{cat}_Recall_fire', 0):.4f}\n")
                            if f'{cat}_AP_smoke' in test_stats:
                                f.write(f"    Smoke - AP: {test_stats.get(f'{cat}_AP_smoke', 0):.4f} | Recall: {test_stats.get(f'{cat}_Recall_smoke', 0):.4f}\n")
