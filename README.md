# Hot Topic

Use python satellite imagery, historical weather data, and historical fire perimeters to predict wildfire spreading using artificial neural network.

# Requirements
Install required libraries using the following command in the root directory:
```bash
pip install -r requirements.txt
```
You may have to create a virtual environment. Use the following commands:
```bash
python -m venv myvenv
source myvenv/bin/activate
```
The first command creates a virtual environment called myvenv and the second command can be used 
to activate it.

## Scripts Overview

This project includes several key scripts for wildfire prediction and data visualization:

### 1. runProduction.py - Production Fire Prediction

**Purpose**: Run fire predictions in production mode (no ground truth required) and generate visualizations and GeoJSON outputs.

**Key Features**:
- Predicts fire spread without requiring next-day perimeter data
- Generates probability maps with predicted fire perimeters
- Creates GeoJSON files with fire perimeter polygons
- Saves visualization images showing probability maps with perimeter overlays
- Handles multiple fires and dates automatically
- Comprehensive error handling and logging

**Usage**:

```bash
# Run predictions on all fires in training_data
python3.10 runProduction.py

# Run predictions on all fires (explicit)
python3.10 runProduction.py --all

# Run prediction for specific fire and date
python3.10 runProduction.py --fire <fire_name> --date <MMDD>

# Run with custom parameters
python3.10 runProduction.py --fire <fire_name> --date <MMDD> --points 15000 --model 20200903-193223mod
```

**Arguments**:
- `--fire`: Fire name (as in training_data folder)
- `--date`: Date in MMDD format (e.g., 0725)
- `--points`: Number of points to sample (default: 10000)
- `--model`: Model file name without .h5 extension (default: 20200903-193223mod)
- `--all`: Run predictions on all fires in training_data

**Outputs**:
- **Log files**: `output/runProduction_YYYYMMDD_HHMMSS.log` and `output/runProduction_YYYYMMDD_HHMMSS_errors.log`
- **Visualization images**: `output/images/perimeter_viz_[FIRE]_[DATE].png`
- **GeoJSON file**: `output/all_predictions.geojson` (if center.json files exist)
- **CSV files**: `output/predictions_[FIRE]_[DATE].csv` (single fire mode)

**Example**:
```bash
python3.10 runProduction.py --fire EAST_CREEK --date 0725
```

### 2. runPrediction.py - Evaluation Fire Prediction

**Purpose**: Run fire predictions with ground truth comparison and performance metrics.

**Key Features**:
- Evaluates prediction accuracy against actual fire perimeters
- Generates performance metrics and visualizations
- Can run in evaluation mode (with ground truth) or inference mode (without)
- Creates detailed visualizations with probability-based coloring

**Usage**:

```bash
# List available fires and dates
python3.10 runPrediction.py --list

# Run evaluation for specific fire and date (with ground truth)
python3.10 runPrediction.py --fire <fire_name> --date <MMDD> --eval

# Run inference for specific fire and date (no ground truth)
python3.10 runPrediction.py --fire <fire_name> --date <MMDD> --no-eval

# Run with custom parameters
python3.10 runPrediction.py --fire <fire_name> --date <MMDD> --points 15000 --eval
```

**Arguments**:
- `--fire`: Fire name (as in data folder)
- `--date`: Date in MMDD format (e.g., 0711)
- `--points`: Number of points to sample (default: 10000)
- `--list`: List available fires and dates
- `--eval`: Run in evaluation mode (requires next day perimeter, computes metrics)
- `--no-eval`: Run in inference/production mode (no metrics, no next day perimeter required)

**Outputs**:
- **Visualization images**: `output/figures/[FIRE]_[DATE]_radius[radius]_points[count].png`
- **Performance metrics**: Printed to console

**Example**:
```bash
python3.10 runPrediction.py --fire beaverCreek --date 0711 --eval
```

### 3. view_npy.py - NPY File Viewer

**Purpose**: View and analyze NPY (NumPy) files containing terrain data, satellite imagery, and fire perimeters.

**Key Features**:
- Visualize terrain data (DEM, slope, aspect, NDVI)
- View Landsat satellite bands
- Display fire perimeter data
- Automatic colormap selection based on data type
- Statistical information display
- Support for multi-channel images

**Usage**:

```bash
# List available NPY files
python3.10 view_npy.py --list

# View terrain data for specific fire
python3.10 view_npy.py --fire <fire_name> --type <data_type>

# View perimeter data for specific fire and date
python3.10 view_npy.py --fire <fire_name> --date <MMDD>

# View madre perimeter data
python3.10 view_npy.py --madre <MMDD>

# View specific file with custom colormap
python3.10 view_npy.py --file <path_to_file> --cmap <colormap>
```

**Arguments**:
- `--file`: Direct path to NPY file
- `--fire`: Fire name (e.g., beaverCreek)
- `--type`: Type of terrain data (dem, aspect, slope, ndvi, band_2, band_3, band_4, band_5)
- `--date`: Date for perimeter data (e.g., 0711)
- `--madre`: Date for madre perimeter data (e.g., 0707)
- `--list`: List available NPY files
- `--cmap`: Colormap to use (e.g., gray, terrain, plasma, hsv, RdYlGn, Reds)

**Available Data Types**:
- `dem`: Digital Elevation Model (terrain colormap)
- `slope`: Slope data (plasma colormap)
- `aspect`: Aspect data (hsv colormap)
- `ndvi`: Normalized Difference Vegetation Index (RdYlGn colormap)
- `band_2`, `band_3`, `band_4`, `band_5`: Landsat satellite bands (gray colormap)

**Example**:
```bash
# View DEM data
python3.10 view_npy.py --fire beaverCreek --type dem

# View NDVI with custom colormap
python3.10 view_npy.py --fire beaverCreek --type ndvi --cmap RdYlGn

# View fire perimeter
python3.10 view_npy.py --fire beaverCreek --date 0711
```

## Data Directory Structure

### Training Data (`/training_data`)
Used by `runProduction.py` for production predictions:

```
training_data/
├── [FIRE_NAME]/
│   ├── center.json          # Geographic center coordinates (required for GeoJSON output)
│   ├── dem.npy              # Digital Elevation Model
│   ├── aspect.npy           # Terrain aspect
│   ├── slope.npy            # Terrain slope
│   ├── ndvi.npy             # Vegetation index
│   ├── band_2.npy           # Landsat band 2
│   ├── band_3.npy           # Landsat band 3
│   ├── band_4.npy           # Landsat band 4
│   ├── band_5.npy           # Landsat band 5
│   ├── weather/             # Weather data files
│   └── perims/              # Fire perimeter data
│       ├── 0711.npy         # Perimeter for July 11
│       ├── 0712.npy         # Perimeter for July 12
│       └── ...
```

### Evaluation Data (`/data`)
Used by `runPrediction.py` for evaluation with ground truth:

```
data/
├── [FIRE_NAME]/
│   ├── dem.npy              # Digital Elevation Model
│   ├── aspect.npy           # Terrain aspect
│   ├── slope.npy            # Terrain slope
│   ├── ndvi.npy             # Vegetation index
│   ├── band_2.npy           # Landsat band 2
│   ├── band_3.npy           # Landsat band 3
│   ├── band_4.npy           # Landsat band 4
│   ├── band_5.npy           # Landsat band 5
│   ├── weather/             # Weather data files
│   └── perims/              # Fire perimeter data
│       ├── 0711.npy         # Perimeter for July 11
│       ├── 0712.npy         # Perimeter for July 12
│       └── ...
```

**Note**: The model is currently configured to use `/training_data`. To use `/data` for evaluation, the model files need to be modified.

## Output Directory Structure

```
output/
├── runProduction_YYYYMMDD_HHMMSS.log      # Main execution log
├── runProduction_YYYYMMDD_HHMMSS_errors.log # Error log (TensorFlow warnings, etc.)
├── all_predictions.geojson                # All predicted fire perimeters (GeoJSON format)
├── images/                                # Predicted perimeter visualization images
│   ├── perimeter_viz_[FIRE]_[DATE].png    # Probability maps with predicted perimeters
│   └── ...
├── figures/                               # Evaluation visualizations (from runPrediction.py)
│   ├── [FIRE]_[DATE]_radius[radius]_points[count].png
│   └── ...
└── datasets/                              # Legacy output from older FireCast model
    ├── test03Sep/
    ├── train03Sep/
    └── validate03Sep/
```

### Output Files Description

#### Log Files
- **`runProduction_YYYYMMDD_HHMMSS.log`**: Main execution log containing:
  - Fire processing status
  - Data validation results
  - Prediction generation progress
  - Error messages and warnings
- **`runProduction_YYYYMMDD_HHMMSS_errors.log`**: Error-specific log containing:
  - TensorFlow/Keras warnings
  - Model loading messages
  - System-level errors

#### Prediction Outputs
- **`all_predictions.geojson`**: Combined GeoJSON file containing:
  - Predicted fire perimeter polygons for all processed fires
  - Properties including fire name, date, average prediction probability
  - Only generated if `center.json` files exist in fire directories
- **`perimeter_viz_[FIRE]_[DATE].png`**: Visualization images showing:
  - Probability heatmap (0-1 scale)
  - Predicted fire perimeter (cyan line)
  - Statistics (burned pixels, burn area percentage, average probability)

#### Legacy Outputs
- **`datasets/`**: Contains training/validation/test splits from the original FireCast model
- **`figures/`**: Evaluation visualizations from `runPrediction.py` (when using ground truth data)

## Troubleshooting

### Common Issues

1. **"zero-size array to reduction operation maximum"**: This error occurs when data layers contain all NaN values. Check that your terrain data files are properly formatted.

2. **"Missing center.json"**: Each fire directory needs a `center.json` file with `center_lat` and `center_lon` coordinates for GeoJSON output.

3. **"No vulnerable pixels found"**: This happens when the starting perimeter is empty or outside the image bounds. Check your perimeter files.

4. **Model loading warnings**: Keras warnings about optimizer state are normal and don't affect functionality.

### Data Validation

Use `view_npy.py` to inspect your data files:
```bash
# Check if DEM data is valid
python3.10 view_npy.py --fire <fire_name> --type dem

# Verify perimeter data exists
python3.10 view_npy.py --fire <fire_name> --date <MMDD>
```

### Performance Tips

- Use `--points` to control memory usage (lower values = less memory)
- Run production mode for large-scale predictions
- Use evaluation mode only when you have ground truth data
- Check log files in `output/` directory for detailed error information
