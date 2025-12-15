# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-precision farmland identification system aiming for 0.5-meter accuracy in boundary detection. The system processes aerial TIF images and uses AI/deep learning techniques to identify cultivated land areas and track changes over time.

## Key Components

### Core Modules
- **耕地分析系统.py** - Main analysis engine with TIF processing, geospatial handling, and model inference
- **耕地分析工具_图形界面.py** - Tkinter GUI application for easy operation
- **耕地识别模型训练(16)(1).py** - Model training script for U-Net based segmentation
- **高精度颜色识别.py** - Advanced color recognition module (multi-method fusion)
- **耕地变化评价指标.py** - Professional evaluation tools for farmland change assessment

### Data Flow
1. Input TIF images (with geospatial coordinates)
2. Optional: Shapefile annotations for supervised learning
3. Model prediction using U-Net or high-precision color recognition
4. Boundary accuracy assessment and area change analysis
5. Generate reports with professional metrics

## Common Commands

### Running the System
```bash
# GUI application (recommended for most users)
python 耕地分析工具_图形界面.py

# Command-line analysis
python 耕地分析系统.py

# Model training
python 耕地识别模型训练(16)(1).py

# Test high-precision recognition
python 高精度颜色识别.py

# Deploy check
python 部署检查.py
```

### Progress Management
```bash
# Record progress
python progress_manager.py

# View progress
cat docs/PROGRESS.md
```

## Architecture

### High-Precision Recognition Pipeline
The system uses a multi-method fusion approach:
1. **Adaptive Color Thresholding** - Dynamically adjusts based on local statistics
2. **K-means Color Clustering** - Identifies farmland color patterns
3. **LAB Space Enhancement** - Improves detection in CIELAB color space

### Intelligent Incremental Prediction
- Only processes boundary regions (typically 2-3% of total area)
- Preserves stable areas from previous year's data
- Achieves 38.9x speedup compared to full image processing

### Boundary Accuracy Assessment
- Provides 5-tier precision levels (S/A/B/C/D)
- Measures average boundary offset in meters
- Validates against 0.5-meter accuracy requirement

## Development Guidelines

### Image Requirements
- Resolution: ≤0.5 meters/pixel for best results
- Format: GeoTIFF with proper CRS information
- Recommended: Drone imagery (5-10cm resolution) or high-res satellite (≤0.8m)

### Dependencies
- Core: numpy, rasterio, geopandas, opencv-python
- ML: tensorflow, scikit-learn
- GUI: tkinter (built-in), Pillow

### File Organization
- Main modules in root directory
- `docs/` - Progress tracking and documentation
- Model files: `.h5` for TensorFlow models
- Cache: `.pkl` files for preprocessed data

### Performance Optimization
- Use GPU acceleration when available (auto-detected)
- Enable `使用高精度增强` flag for improved training data
- Adjust block size based on available memory

## Testing and Validation

Run the deploy checker to validate the installation:
```bash
python 部署检查.py
```

This verifies:
- All required files exist
- Dependencies are installed
- High-precision recognition module works correctly

## Important Implementation Details

### Configuration
Edit the configuration section in `耕地分析系统.py` (lines 11-41) to set:
- Input/output paths
- Block size and overlap parameters
- Analysis mode selection
- Model paths

### Year-over-Year Analysis
The system supports intelligent comparison between years:
- Loads baseline data from previous year's analysis
- Performs incremental prediction on boundary regions only
- Generates professional evaluation reports with boundary offset metrics

### Error Handling
- Automatic fallback from high-precision to basic color recognition
- Graceful degradation when GPU unavailable
- Comprehensive logging for debugging

## Recent Updates

- Fixed OpenCV compatibility issues in high-precision color recognition
- Simplified evaluation metrics (removed complex "垄" concepts)
- Added comprehensive progress tracking system
- Improved boundary accuracy calculation methods