#!/bin/bash

# NeRF Training Quick Start Script
# This script helps you set up and train NeRF on the LEGO dataset

set -e  # Exit on error

echo "======================================"
echo "NeRF Training Quick Start"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Python
echo "Checking Python installation..."
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.7+"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_status "Python $PYTHON_VERSION found"
echo ""

# Check for CUDA
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo ""

# Check directory structure
echo "Checking directory structure..."

if [ ! -d "data/nerf_synthetic/lego" ]; then
    print_error "LEGO dataset not found at data/nerf_synthetic/lego"
    echo "Please move your dataset to: data/nerf_synthetic/lego"
    exit 1
fi
print_status "LEGO dataset found"

if [ ! -f "data/nerf_synthetic/lego/transforms_train.json" ]; then
    print_error "transforms_train.json not found"
    exit 1
fi
print_status "Dataset files verified"

# Create directories
mkdir -p configs
mkdir -p logs
print_status "Created necessary directories"
echo ""

# Count images
TRAIN_IMAGES=$(ls data/nerf_synthetic/lego/train/*.png 2>/dev/null | wc -l)
TEST_IMAGES=$(ls data/nerf_synthetic/lego/test/*.png 2>/dev/null | wc -l)
VAL_IMAGES=$(ls data/nerf_synthetic/lego/val/*.png 2>/dev/null | wc -l)

echo "Dataset statistics:"
echo "  Training images: $TRAIN_IMAGES"
echo "  Test images: $TEST_IMAGES"
echo "  Validation images: $VAL_IMAGES"
echo ""

# Ask user if they want to proceed
read -p "Ready to start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Start training
echo ""
echo "======================================"
echo "Starting NeRF Training"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Dataset: LEGO"
echo "  Iterations: 200,000"
echo "  Log directory: ./logs/lego_metrics"
echo ""
echo "Estimated time: 4-12 hours (depending on GPU)"
echo ""
print_warning "This will take several hours. Consider using screen/tmux!"
echo ""

# Run training with metrics
python run_nerf_with_metrics.py --config configs/lego_config.txt

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    print_status "Training completed successfully!"
    echo "======================================"
    echo ""
    
    # Analyze results
    echo "Analyzing results..."
    python analyze_results.py \
        --log_dir logs/lego_metrics \
        --save_plot logs/lego_metrics/detailed_analysis.png \
        --export_table logs/lego_metrics/results_table.md
    
    echo ""
    echo "Results saved to: logs/lego_metrics/"
    echo "  - training_metrics.npz"
    echo "  - test_metrics.npz"
    echo "  - summary_metrics.json"
    echo "  - training_metrics.png"
    echo "  - detailed_analysis.png"
    echo "  - results_table.md"
    echo ""
    
else
    echo ""
    print_error "Training failed. Check logs for details."
    exit 1
fi
