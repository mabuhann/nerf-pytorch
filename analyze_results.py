"""
Analyze NeRF Training Results
This script loads and analyzes the metrics from NeRF training.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse


def load_metrics(log_dir):
    """Load all metrics from the log directory"""
    metrics = {}
    
    # Load training metrics
    train_file = os.path.join(log_dir, 'training_metrics.npz')
    if os.path.exists(train_file):
        train_data = np.load(train_file)
        metrics['train'] = {
            'iterations': train_data['iterations'],
            'losses': train_data['losses'],
            'psnrs': train_data['psnrs'],
            'step_times': train_data['step_times']
        }
        print(f"✓ Loaded training metrics: {len(train_data['iterations'])} iterations")
    else:
        print(f"✗ Training metrics not found at {train_file}")
    
    # Load test metrics
    test_file = os.path.join(log_dir, 'test_metrics.npz')
    if os.path.exists(test_file):
        test_data = np.load(test_file)
        metrics['test'] = {
            'iterations': test_data['iterations'],
            'psnrs': test_data['psnrs'],
        }
        if 'ssims' in test_data and test_data['ssims'] is not None:
            metrics['test']['ssims'] = test_data['ssims']
        print(f"✓ Loaded test metrics: {len(test_data['iterations'])} evaluations")
    else:
        print(f"✗ Test metrics not found at {test_file}")
    
    # Load summary
    summary_file = os.path.join(log_dir, 'summary_metrics.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            metrics['summary'] = json.load(f)
        print(f"✓ Loaded summary metrics")
    else:
        print(f"✗ Summary not found at {summary_file}")
    
    return metrics


def print_summary(metrics):
    """Print a formatted summary of the training"""
    if 'summary' not in metrics:
        print("No summary available")
        return
    
    summary = metrics['summary']
    
    print("\n" + "="*70)
    print("NERF TRAINING SUMMARY")
    print("="*70)
    
    # Training time
    if 'total_training_time_hours' in summary:
        hours = summary['total_training_time_hours']
        print(f"\n📅 Training Time: {hours:.2f} hours ({hours*60:.1f} minutes)")
    
    # Training metrics
    print(f"\n🎯 Training Metrics:")
    if 'final_train_psnr' in summary and summary['final_train_psnr']:
        print(f"   Final Train PSNR: {summary['final_train_psnr']:.2f} dB")
    if 'best_train_psnr' in summary and summary['best_train_psnr']:
        print(f"   Best Train PSNR:  {summary['best_train_psnr']:.2f} dB")
    if 'final_train_loss' in summary and summary['final_train_loss']:
        print(f"   Final Train Loss: {summary['final_train_loss']:.6f}")
    
    # Test metrics
    print(f"\n🧪 Test Set Metrics:")
    if 'final_test_psnr' in summary and summary['final_test_psnr']:
        print(f"   Final Test PSNR: {summary['final_test_psnr']:.2f} dB")
    if 'best_test_psnr' in summary and summary['best_test_psnr']:
        print(f"   Best Test PSNR:  {summary['best_test_psnr']:.2f} dB")
    if 'avg_test_ssim' in summary and summary['avg_test_ssim']:
        print(f"   Average SSIM:    {summary['avg_test_ssim']:.4f}")
    
    # Performance
    print(f"\n⚡ Performance:")
    if 'avg_step_time' in summary and summary['avg_step_time']:
        avg_time = summary['avg_step_time']
        print(f"   Avg Step Time: {avg_time:.4f} seconds")
        print(f"   Steps per hour: {int(3600 / avg_time)}")
    
    if 'total_iterations' in summary:
        print(f"   Total Iterations: {summary['total_iterations']}")
    
    print("="*70 + "\n")


def plot_detailed_analysis(metrics, save_path=None):
    """Create detailed plots of training metrics"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training Loss
    if 'train' in metrics:
        ax1 = plt.subplot(3, 3, 1)
        iterations = metrics['train']['iterations']
        losses = metrics['train']['losses']
        ax1.plot(iterations, losses, linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Training PSNR
        ax2 = plt.subplot(3, 3, 2)
        psnrs = metrics['train']['psnrs']
        ax2.plot(iterations, psnrs, linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Training PSNR')
        ax2.grid(True, alpha=0.3)
        
        # 3. Step Time
        ax3 = plt.subplot(3, 3, 3)
        step_times = metrics['train']['step_times']
        ax3.plot(iterations, step_times, linewidth=0.5, alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Time per Training Step')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss (smoothed)
        ax4 = plt.subplot(3, 3, 4)
        window = 100
        if len(losses) > window:
            smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_iters = iterations[window-1:]
            ax4.plot(smoothed_iters, smoothed_loss, linewidth=1.5)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss (smoothed)')
        ax4.set_title(f'Training Loss (smoothed, window={window})')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # 5. PSNR (smoothed)
        ax5 = plt.subplot(3, 3, 5)
        if len(psnrs) > window:
            smoothed_psnr = np.convolve(psnrs, np.ones(window)/window, mode='valid')
            smoothed_iters = iterations[window-1:]
            ax5.plot(smoothed_iters, smoothed_psnr, linewidth=1.5)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('PSNR (dB, smoothed)')
        ax5.set_title(f'Training PSNR (smoothed, window={window})')
        ax5.grid(True, alpha=0.3)
        
        # 6. Step Time Distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(step_times, bins=50, alpha=0.7, edgecolor='black')
        ax6.axvline(np.mean(step_times), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(step_times):.4f}s')
        ax6.axvline(np.median(step_times), color='g', linestyle='--',
                   label=f'Median: {np.median(step_times):.4f}s')
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Step Time Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Test PSNR
    if 'test' in metrics:
        ax7 = plt.subplot(3, 3, 7)
        test_iters = metrics['test']['iterations']
        test_psnrs = metrics['test']['psnrs']
        ax7.plot(test_iters, test_psnrs, marker='o', markersize=8, linewidth=2)
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('PSNR (dB)')
        ax7.set_title('Test Set PSNR')
        ax7.grid(True, alpha=0.3)
        
        # Add max line
        max_psnr = np.max(test_psnrs)
        ax7.axhline(max_psnr, color='r', linestyle='--', alpha=0.5,
                   label=f'Best: {max_psnr:.2f} dB')
        ax7.legend()
        
        # 8. Test SSIM (if available)
        if 'ssims' in metrics['test']:
            ax8 = plt.subplot(3, 3, 8)
            test_ssims = metrics['test']['ssims']
            ax8.plot(test_iters, test_ssims, marker='s', markersize=8, 
                    linewidth=2, color='green')
            ax8.set_xlabel('Iteration')
            ax8.set_ylabel('SSIM')
            ax8.set_title('Test Set SSIM')
            ax8.grid(True, alpha=0.3)
            ax8.set_ylim([0.9, 1.0])
            
            # Add average line
            avg_ssim = np.mean(test_ssims)
            ax8.axhline(avg_ssim, color='r', linestyle='--', alpha=0.5,
                       label=f'Avg: {avg_ssim:.4f}')
            ax8.legend()
    
    # 9. Training progress (percentage complete)
    if 'train' in metrics:
        ax9 = plt.subplot(3, 3, 9)
        total_iters = iterations[-1]
        progress = (iterations / total_iters) * 100
        ax9.plot(progress, psnrs, linewidth=1, alpha=0.7)
        ax9.set_xlabel('Training Progress (%)')
        ax9.set_ylabel('PSNR (dB)')
        ax9.set_title('PSNR vs Training Progress')
        ax9.grid(True, alpha=0.3)
        ax9.set_xlim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed analysis plot to {save_path}")
    
    return fig


def generate_report_table(metrics):
    """Generate a markdown table for the report"""
    if 'summary' not in metrics:
        return "No summary available"
    
    summary = metrics['summary']
    
    table = """
## NeRF Training Results

| Metric | Value |
|--------|-------|
"""
    
    if 'total_training_time_hours' in summary:
        table += f"| Training Time | {summary['total_training_time_hours']:.2f} hours |\n"
    
    if 'best_train_psnr' in summary and summary['best_train_psnr']:
        table += f"| Best Training PSNR | {summary['best_train_psnr']:.2f} dB |\n"
    
    if 'final_train_psnr' in summary and summary['final_train_psnr']:
        table += f"| Final Training PSNR | {summary['final_train_psnr']:.2f} dB |\n"
    
    if 'best_test_psnr' in summary and summary['best_test_psnr']:
        table += f"| Best Test PSNR | {summary['best_test_psnr']:.2f} dB |\n"
    
    if 'final_test_psnr' in summary and summary['final_test_psnr']:
        table += f"| Final Test PSNR | {summary['final_test_psnr']:.2f} dB |\n"
    
    if 'avg_test_ssim' in summary and summary['avg_test_ssim']:
        table += f"| Average Test SSIM | {summary['avg_test_ssim']:.4f} |\n"
    
    if 'avg_step_time' in summary and summary['avg_step_time']:
        table += f"| Average Step Time | {summary['avg_step_time']:.4f} seconds |\n"
    
    if 'total_iterations' in summary:
        table += f"| Total Iterations | {summary['total_iterations']:,} |\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description='Analyze NeRF training results')
    parser.add_argument('--log_dir', type=str, default='./logs/lego_metrics',
                       help='Path to the log directory')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save the analysis plot')
    parser.add_argument('--show_plot', action='store_true',
                       help='Display the plot')
    parser.add_argument('--export_table', type=str, default=None,
                       help='Export markdown table to file')
    
    args = parser.parse_args()
    
    print(f"\n🔍 Analyzing NeRF training results from: {args.log_dir}\n")
    
    # Load metrics
    metrics = load_metrics(args.log_dir)
    
    if not metrics:
        print("❌ No metrics found!")
        return
    
    # Print summary
    print_summary(metrics)
    
    # Generate report table
    if args.export_table:
        table = generate_report_table(metrics)
        with open(args.export_table, 'w') as f:
            f.write(table)
        print(f"✓ Exported markdown table to {args.export_table}\n")
    
    # Create plots
    if args.save_plot or args.show_plot:
        print("📊 Generating detailed analysis plots...")
        fig = plot_detailed_analysis(metrics, save_path=args.save_plot)
        
        if args.show_plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == '__main__':
    main()
