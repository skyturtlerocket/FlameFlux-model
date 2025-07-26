#!/usr/bin/env python3.10
import tensorflow as tf
from lib import viz
from math import cos, radians
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import sys
import os
import random
import argparse
import csv
from lib import rawdata
from lib import dataset
from lib import model
from lib import preprocess
import json
from scipy.ndimage import binary_fill_holes
import alphashape
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from scipy.spatial import distance
from concave_hull import concave_hull
import cv2

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Redirect stdout and stderr to output files
import sys
from datetime import datetime

# Create timestamp for unique log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"output/runProduction_{timestamp}.log"
error_file = f"output/runProduction_{timestamp}_errors.log"

# Redirect stdout to log file
sys.stdout = open(log_file, 'w')
sys.stderr = open(error_file, 'w')

print(f"Starting runProduction.py at {datetime.now()}")
print(f"Log file: {log_file}")
print(f"Error file: {error_file}")


def run_production_inference(fire_name, date, target_points=10000, model_file="20200903-193223mod", all_predictions=None):
    print(f"[Production] Loading fire data for {fire_name} on date {date}...")
    
    try:
        # Load data for the specified fire and date in inference mode
        data = rawdata.RawData.load(burnNames=[fire_name], dates={fire_name: [date]}, inference=True)
        
        # Validate that we have valid data
        if not data or not data.burns or fire_name not in data.burns:
            print(f"Warning: No valid data found for fire {fire_name}")
            return None
            
        burn = data.burns[fire_name]
        if not burn.layers or len(burn.layers) == 0:
            print(f"Warning: No layers found for fire {fire_name}")
            return None
            
        # Debug: Check layer shapes and content
        print(f"Debug: Fire {fire_name} has {len(burn.layers)} layers")
        for layer_name, layer_data in burn.layers.items():
            valid_count = np.sum(~np.isnan(layer_data))
            print(f"  {layer_name}: shape {layer_data.shape}, valid values: {valid_count}")
            if valid_count == 0:
                print(f"    WARNING: Layer {layer_name} has no valid values!")
                # Debug the problematic layer
                print(f"    Debug {layer_name}: min={np.min(layer_data)}, max={np.max(layer_data)}")
                print(f"    Debug {layer_name}: unique values: {np.unique(layer_data)[:10]}")  # First 10 unique values
                print(f"    Debug {layer_name}: NaN count: {np.sum(np.isnan(layer_data))}")
                print(f"    Debug {layer_name}: Inf count: {np.sum(np.isinf(layer_data))}")
                return None
            
        # Create dataset with vulnerable pixels around fire perimeter
        test_dataset = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
        print(f"Created dataset with {len(test_dataset)} total points")
        
        # Debug: Check perimeter data
        if len(test_dataset) == 0:
            print(f"Debug: Checking why no vulnerable pixels found for {fire_name}")
            day = data.getDay(fire_name, date)
            print(f"  Starting perimeter shape: {day.startingPerim.shape}")
            print(f"  Starting perimeter sum: {np.sum(day.startingPerim)}")
            print(f"  Starting perimeter min/max: {np.min(day.startingPerim)}/{np.max(day.startingPerim)}")
            print(f"  Starting perimeter unique values: {np.unique(day.startingPerim)}")
            
            # Check if perimeter file exists
            perim_path = f"training_data/{fire_name}/perims/{date}.npy"
            print(f"  Perimeter file exists: {os.path.exists(perim_path)}")
            
            # Try to understand the vulnerable pixels calculation
            startingPerim = day.startingPerim
            kernel = np.ones((3,3))
            radius = dataset.Dataset.VULNERABLE_RADIUS
            its = int(round((2*(radius)**2)**.5))
            print(f"  Dilating with radius {radius}, iterations {its}")
            dilated = cv2.dilate(startingPerim, kernel, iterations=its)
            border = dilated - startingPerim
            print(f"  Dilated sum: {np.sum(dilated)}")
            print(f"  Border sum: {np.sum(border)}")
            ys, xs = np.where(border)
            print(f"  Border pixels found: {len(ys)}")
        
        # Check if we have any valid points
        if len(test_dataset) == 0:
            print(f"Warning: No vulnerable pixels found for fire {fire_name} on date {date}")
            return None
            
        # Sample points to reduce density while maintaining coverage
        all_points = test_dataset.toList(test_dataset.points)
        if len(all_points) > target_points:
            print(f"Sampling {target_points} points from {len(all_points)} total points (balanced sampling)")
            sampled_points = random.sample(all_points, target_points)
            test_dataset = dataset.Dataset(data, sampled_points)
            
    except Exception as e:
        print(f"Error loading data for {fire_name} {date}: {e}")
        return None
    
    # Load model and preprocessor
    try:
        mod, pp = getModel(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    # Preprocess in inference mode (returns only inputs and ptList)
    try:
        inputs, ptList = pp.process(test_dataset, inference=True)
        
        # Validate inputs
        if not inputs or len(inputs) != 2:
            print(f"Error: Invalid inputs generated for {fire_name}")
            return None
            
        weather_inputs, img_inputs = inputs
        if len(weather_inputs) == 0 or len(img_inputs) == 0:
            print(f"Error: Empty input arrays for {fire_name}")
            return None
            
    except Exception as e:
        print(f"Error processing data for {fire_name} {date}: {e}")
        return None
    
    # Predict
    predictions = None
    try:
        predictions = mod.predict(inputs).flatten()
        print(f"Generated {len(predictions)} predictions")
        
        # Create perimeter visualization
        if all_predictions is None:  # Only for single fire mode
            # Ensure the images directory exists
            os.makedirs("output/images", exist_ok=True)
            viz_path = f"output/images/perimeter_viz_{fire_name}_{date}.png"
            create_perimeter_visualization(predictions, ptList, fire_name, date, None, viz_path)
            
    except Exception as e:
        print(f"Error making predictions for {fire_name} {date}: {e}")
        return None
    
    # Ensure predictions were generated successfully
    if predictions is None:
        print(f"Error: No predictions generated for {fire_name} {date}")
        return None
    
    # Store predictions for GeoJSON output if all_predictions dict provided
    if all_predictions is not None:
        # --- Calculate bounding box for this fire/date (same as getData.py) ---
        img_size = (1024, 1024)
        area_m = 1024 * 30
        # Find center_lat, center_lon for this fire
        fire_dir = fire_name.replace(' ', '_')
        meta_path = os.path.join('training_data', fire_dir, 'center.json')
        print(f"Looking for center.json at: {meta_path}")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                center_lat = meta['center_lat']
                center_lon = meta['center_lon']
        else:
            print(f"Warning: Missing center.json for fire {fire_name}, skipping GeoJSON output")
            print(f"Available files in {os.path.join('training_data', fire_dir)}: {os.listdir(os.path.join('training_data', fire_dir)) if os.path.exists(os.path.join('training_data', fire_dir)) else 'Directory not found'}")
            return None
            
        lat_meter = 111320
        lon_meter = 111320 * cos(radians(center_lat))
        half_width_deg = (area_m / 2) / lon_meter
        half_height_deg = (area_m / 2) / lat_meter
        min_lat = center_lat - half_height_deg
        max_lat = center_lat + half_height_deg
        min_lon = center_lon - half_width_deg
        max_lon = center_lon + half_width_deg
        
        def pixel_to_latlon(x, y):
            lon = min_lon + (x / (img_size[1] - 1)) * (max_lon - min_lon)
            lat = max_lat - (y / (img_size[0] - 1)) * (max_lat - min_lat)
            return lat, lon
        
        for pt, pred in zip(ptList, predictions):
            burnName, date, (y, x) = pt
            lat, lon = pixel_to_latlon(x, y)
            key = (fire_name, date, (lat, lon))  # (fire_name, date, (lat, lon))
            all_predictions[key] = pred
        
        # Create perimeter visualization for all_predictions mode
        # Ensure the images directory exists
        os.makedirs("output/images", exist_ok=True)
        viz_path = f"output/images/perimeter_viz_{fire_name}_{date}.png"
        create_perimeter_visualization(predictions, ptList, fire_name, date, pixel_to_latlon, viz_path)
    
    # Output predictions to CSV with lat/lon (only if not in all_predictions mode)
    if all_predictions is None:
        output_csv = f"output/predictions_{fire_name}_{date}.csv"
        # --- Calculate bounding box for this fire/date (same as getData.py) ---
        img_size = (1024, 1024)
        area_m = 1024 * 30
        # Find center_lat, center_lon for this fire
        fire_dir = fire_name.replace(' ', '_')
        meta_path = os.path.join('training_data', fire_dir, 'center.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                center_lat = meta['center_lat']
                center_lon = meta['center_lon']
        else:
            raise RuntimeError(f"Missing center.json for fire {fire_name} in training_data/{fire_dir}/. Please create this file with center_lat and center_lon.")
        lat_meter = 111320
        lon_meter = 111320 * cos(radians(center_lat))
        half_width_deg = (area_m / 2) / lon_meter
        half_height_deg = (area_m / 2) / lat_meter
        min_lat = center_lat - half_height_deg
        max_lat = center_lat + half_height_deg
        min_lon = center_lon - half_width_deg
        max_lon = center_lon + half_width_deg
        def pixel_to_latlon(x, y):
            lon = min_lon + (x / (img_size[1] - 1)) * (max_lon - min_lon)
            lat = max_lat - (y / (img_size[0] - 1)) * (max_lat - min_lat)
            return lat, lon
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['burnName', 'date', 'y', 'x', 'lat', 'lon', 'predicted_prob'])
            for pt, pred in zip(ptList, predictions):
                burnName, date, (y, x) = pt
                lat, lon = pixel_to_latlon(x, y)
                writer.writerow([burnName, date, y, x, lat, lon, pred])
        print(f"Predictions saved to {output_csv}")
        # Print summary
        burned = sum(1 for p in predictions if p > 0.5)
        print(f"Predicted burned pixels (prob > 0.5): {burned} / {len(predictions)}")

        # --- Visualization: Save annotated image ---

        # --- GeoJSON perimeter export function ---
        def save_perimeter_geojson_and_overlay(prob_map, pixel_to_latlon, out_path, overlay_path, threshold=0.5):
            """
            Extracts all outer perimeters of the predicted burned area (holes filled), saves as a GeoJSON (MultiPolygon if needed),
            and overlays the perimeter(s) on the probability map image.
            """
            # Grid-based outer boundary approach
            burned_pixels = np.argwhere((prob_map > threshold).astype(np.uint8))
            if burned_pixels.shape[0] == 0:
                print('No burned pixels found!')
                return
            pixel_coordinates = [(int(x), int(y)) for y, x in burned_pixels]
            pixel_set = set(pixel_coordinates)
            # Step 1: Find boundary pixels
            boundary_pixels = []
            for x, y in pixel_coordinates:
                neighbors = [
                    (x-1, y-1), (x, y-1), (x+1, y-1),
                    (x-1, y),             (x+1, y),
                    (x-1, y+1), (x, y+1), (x+1, y+1)
                ]
                empty_neighbors = sum(1 for neighbor in neighbors if neighbor not in pixel_set)
                if empty_neighbors > 0:
                    boundary_pixels.append((x, y))
            if not boundary_pixels:
                print('No boundary pixels found!')
                return
            # Step 2: For each angle from centroid, keep only the farthest pixel
            center_x = sum(p[0] for p in boundary_pixels) / len(boundary_pixels)
            center_y = sum(p[1] for p in boundary_pixels) / len(boundary_pixels)
            import math
            angle_groups = {}
            for x, y in boundary_pixels:
                angle = math.atan2(y - center_y, x - center_x)
                angle_key = round(angle, 1)  # Adjust precision as needed
                if angle_key not in angle_groups:
                    angle_groups[angle_key] = []
                angle_groups[angle_key].append((x, y))
            outermost = []
            for angle, pixels in angle_groups.items():
                if pixels:
                    farthest = max(pixels, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                    outermost.append(farthest)
            if len(outermost) < 3:
                print('Not enough outermost pixels for a polygon!')
                return
            # Sort outermost points by angle from centroid
            center_x = sum(p[0] for p in outermost) / len(outermost)
            center_y = sum(p[1] for p in outermost) / len(outermost)
            import math
            outermost_sorted = sorted(
                outermost,
                key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x)
            )
            # Ensure the loop is closed
            if outermost_sorted[0] != outermost_sorted[-1]:
                outermost_sorted.append(outermost_sorted[0])
            # Convert to lat/lon
            polygon = [pixel_to_latlon(x, y) for x, y in outermost_sorted]
            geojson_coords = [[lon, lat] for lat, lon in polygon]
            if geojson_coords[0] != geojson_coords[-1]:
                geojson_coords.append(geojson_coords[0])
            geometry = {"type": "Polygon", "coordinates": [geojson_coords]}
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"threshold": threshold, "method": "grid_outer_boundary_sorted"}
                }]
            }
            with open(out_path, 'w') as f:
                json.dump(geojson, f, indent=2, separators=(',', ': '))
            print(f"Perimeter GeoJSON saved to {out_path}")
            # Overlay: plot probability map and draw the sorted grid-based outer boundary
            plt.figure(figsize=(8, 6))
            plt.imshow(prob_map, cmap='hot', interpolation='nearest')
            arr = np.array(outermost_sorted)
            plt.plot(arr[:,0], arr[:,1], color='cyan', linewidth=2)
            plt.colorbar(label='Predicted Burn Probability')
            plt.title('Predicted Burn Probabilities with Sorted Grid-Based Outer Boundary')
            plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Perimeter overlay visualization saved to: {overlay_path}")
        # Render probability map
        prob_maps = viz.renderPredictions(test_dataset, dict(zip(ptList, predictions)), predictions)
        for (burnName, date), prob_map in prob_maps.items():
            plt.figure(figsize=(8, 6))
            plt.imshow(prob_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Predicted Burn Probability')
            plt.title(f"Predicted Burn Probabilities: {burnName} {date}")
            img_path = f"output/predictions_{burnName}_{date}.png"
            plt.savefig(img_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Prediction visualization saved to: {img_path}")
            # --- Save GeoJSON perimeter and overlay image ---
            geojson_path = f"output/predicted_perimeter_{burnName}_{date}.geojson"
            overlay_path = f"output/predicted_perimeter_{burnName}_{date}_overlay.png"
            save_perimeter_geojson_and_overlay(prob_map, pixel_to_latlon, geojson_path, overlay_path, threshold=0.5)
        return output_csv
    
    return None

def getModel(weightsFile=None):
    numWeatherInputs = 8
    usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)
    if weightsFile:
        fname = weightsFile + '.h5'
        mod = tf.keras.models.load_model(fname)
    else:
        mod = model.fireCastModel(pp)
    return mod, pp

def find_outer_perimeter(pixel_coordinates):
    """
    Find the outermost pixels that form the perimeter
    """
    pixel_set = set(pixel_coordinates)
    outer_boundary = []
    
    for x, y in pixel_coordinates:
        # Check if this pixel is on the outer edge
        # A pixel is on outer edge if it has empty space in outward directions
        
        # Check 8 directions around the pixel
        neighbors = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),             (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        
        # Count how many neighbors are NOT fire pixels
        empty_neighbors = sum(1 for neighbor in neighbors if neighbor not in pixel_set)
        
        # If it has empty neighbors, it's a boundary pixel
        if empty_neighbors > 0:
            # Additional check: is it on the OUTER boundary?
            # Calculate distance from centroid to determine if it's outer edge
            outer_boundary.append((x, y))
    
    return outer_boundary

def filter_outermost_pixels(boundary_pixels):
    """
    From boundary pixels, keep only the outermost ones and ensure proper polygon ordering
    """
    if not boundary_pixels:
        return []
    
    # Find centroid
    center_x = sum(p[0] for p in boundary_pixels) / len(boundary_pixels)
    center_y = sum(p[1] for p in boundary_pixels) / len(boundary_pixels)
    
    # Group pixels by angle from center
    import math
    
    angle_groups = {}
    for x, y in boundary_pixels:
        angle = math.atan2(y - center_y, x - center_x)
        # Round angle to group nearby pixels
        angle_key = round(angle, 1)  # Adjust precision as needed
        
        if angle_key not in angle_groups:
            angle_groups[angle_key] = []
        angle_groups[angle_key].append((x, y))
    
    # For each angle group, keep only the pixel farthest from center
    outermost = []
    for angle, pixels in angle_groups.items():
        if pixels:
            # Find pixel farthest from center
            farthest = max(pixels, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            outermost.append(farthest)
    
    # Sort points in counterclockwise order around the centroid
    if len(outermost) >= 3:
        # Calculate angles from centroid
        angles = []
        for x, y in outermost:
            angle = math.atan2(y - center_y, x - center_x)
            angles.append(angle)
        
        # Sort by angle (counterclockwise)
        sorted_pairs = sorted(zip(outermost, angles), key=lambda pair: pair[1])
        outermost = [point for point, angle in sorted_pairs]
        
        # Ensure polygon is closed (first and last points should be the same)
        if outermost[0] != outermost[-1]:
            outermost.append(outermost[0])
    
    return outermost

def save_all_predictions_to_geojson(all_predictions, output_file="output/all_predictions.geojson"):
    """Save all predictions as polygon perimeters to a single GeoJSON file"""
    
    # Group predictions by fire and date
    fire_date_predictions = {}
    for (fire_name, date, (lat, lon)), prediction in all_predictions.items():
        key = (fire_name, date)
        if key not in fire_date_predictions:
            fire_date_predictions[key] = []
        fire_date_predictions[key].append((lat, lon, prediction))
    
    features = []
    
    for (fire_name, date), predictions in fire_date_predictions.items():
        try:
            # Extract coordinates and predictions
            lats = [p[0] for p in predictions]
            lons = [p[1] for p in predictions]
            pred_values = [p[2] for p in predictions]
            
            if len(lats) > 0:
                # Find high-probability pixels (threshold = 0.5)
                threshold = 0.5
                high_prob_indices = [i for i, pred in enumerate(pred_values) if pred > threshold]
                
                if len(high_prob_indices) >= 3:  # Need at least 3 points for a polygon
                    # Get coordinates of high-probability pixels
                    high_prob_coords = [(lons[i], lats[i]) for i in high_prob_indices]
                    
                    # Find outer perimeter
                    outer_boundary = find_outer_perimeter(high_prob_coords)
                    
                    if len(outer_boundary) >= 3:
                        # Filter to outermost pixels
                        outermost = filter_outermost_pixels(outer_boundary)
                        
                        if len(outermost) >= 3:
                            # Points are already sorted counterclockwise and closed in filter_outermost_pixels
                            outermost_sorted = outermost
                            
                            # Calculate average prediction for this fire/date
                            avg_prediction = np.mean(pred_values)
                            
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [outermost_sorted]
                                },
                                "properties": {
                                    "fire_name": fire_name,
                                    "date": date,
                                    "avg_prediction": float(avg_prediction),
                                    "num_pixels": len(predictions),
                                    "num_high_prob": len(high_prob_indices),
                                    "perimeter_points": len(outermost_sorted),
                                    "threshold": threshold
                                }
                            }
                            features.append(feature)
                        else:
                            print(f"Warning: Not enough outermost pixels for {fire_name} {date} ({len(outermost)} found)")
                    else:
                        print(f"Warning: Not enough boundary pixels for {fire_name} {date} ({len(outer_boundary)} found)")
                else:
                    print(f"Warning: Not enough high-probability pixels for {fire_name} {date} ({len(high_prob_indices)} found)")
                
        except Exception as e:
            print(f"Error creating polygon for {fire_name} {date}: {e}")
            continue
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2, separators=(',', ': '))
    
    print(f"Saved {len(features)} fire perimeter polygons to {output_file}")
    print(f"Note: These are actual fire perimeter polygons based on high-probability predictions.")

def run_all_fires(target_points=10000, model_file="20200903-193223mod"):
    """Run predictions on all fires in the training data"""
    print("Running predictions on all fires in training data...")
    
    # Get all fires from training_data directory
    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    print(f"Found {len(fires)} fires: {fires}")
    
    all_predictions = {}
    
    for fire_name in fires:
        try:
            # Get all available dates for this fire
            dates = rawdata.Day.allGoodDays(fire_name, inference=True)
            print(f"Processing {fire_name} with {len(dates)} dates: {dates}")
            
            for date in dates:
                try:
                    run_production_inference(fire_name, date, target_points, model_file, all_predictions)
                except Exception as e:
                    print(f"Error processing {fire_name} {date}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing fire {fire_name}: {e}")
            continue
    
    # Save all predictions to GeoJSON
    if all_predictions:
        save_all_predictions_to_geojson(all_predictions)
    else:
        print("No predictions generated")
    
    return all_predictions

def create_perimeter_visualization(predictions, ptList, fire_name, date, pixel_to_latlon, output_path):
    """
    Create a visualization showing probability map with predicted perimeter line overlaid
    """
    # Create a 1024x1024 probability map
    prob_map = np.zeros((1024, 1024), dtype=np.float32)
    
    # Fill in the probability values
    for pt, pred in zip(ptList, predictions):
        burnName, date, (y, x) = pt
        prob_map[y, x] = pred
    
    # Find high-probability pixels for perimeter
    threshold = 0.5
    high_prob_mask = prob_map > threshold
    
    if np.sum(high_prob_mask) > 0:
        # Get coordinates of high-probability pixels
        high_prob_coords = list(zip(*np.where(high_prob_mask)))
        high_prob_coords = [(int(x), int(y)) for y, x in high_prob_coords]
        
        # Find outer perimeter
        outer_boundary = find_outer_perimeter(high_prob_coords)
        
        if len(outer_boundary) >= 3:
            # Filter to outermost pixels
            outermost = filter_outermost_pixels(outer_boundary)
            
            if len(outermost) >= 3:
                # Create the visualization
                plt.figure(figsize=(12, 10))
                
                # Plot probability map
                plt.imshow(prob_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
                plt.colorbar(label='Predicted Burn Probability')
                
                # Plot perimeter line
                if len(outermost) > 1:
                    # Convert to arrays for plotting
                    x_coords = [p[0] for p in outermost]
                    y_coords = [p[1] for p in outermost]
                    
                    # Plot the perimeter line
                    plt.plot(x_coords, y_coords, color='cyan', linewidth=3, label='Predicted Perimeter')
                    plt.plot(x_coords, y_coords, color='blue', linewidth=1, alpha=0.8)
                
                # Add title and labels
                plt.title(f'Fire Prediction: {fire_name} ({date})\nProbability Map with Predicted Perimeter', fontsize=14)
                plt.xlabel('Pixel X Coordinate')
                plt.ylabel('Pixel Y Coordinate')
                plt.legend()
                
                # Add statistics
                burned_pixels = np.sum(high_prob_mask)
                total_pixels = prob_map.size
                burn_percentage = (burned_pixels / total_pixels) * 100
                avg_prob = np.mean(predictions)
                
                stats_text = f'Burned Pixels: {burned_pixels:,}\nBurn Area: {burn_percentage:.2f}%\nAvg Probability: {avg_prob:.3f}'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Perimeter visualization saved to: {output_path}")
                return True
            else:
                print(f"Warning: Not enough outermost pixels for visualization of {fire_name} {date}")
        else:
            print(f"Warning: Not enough boundary pixels for visualization of {fire_name} {date}")
    else:
        print(f"Warning: No high-probability pixels found for visualization of {fire_name} {date}")
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run production fire prediction (no ground truth required)')
    parser.add_argument('--fire', type=str, help='Fire name (as in data folder)')
    parser.add_argument('--date', type=str, help='Date in MMDD format')
    parser.add_argument('--points', type=int, default=10000, help='Number of points to sample (default: 10000)')
    parser.add_argument('--model', type=str, default="20200903-193223mod", help='Model file name (without .h5)')
    parser.add_argument('--all', action='store_true', help='Run predictions on all fires in training data')
    args = parser.parse_args()

    # If no arguments provided, run on all fires
    if len(sys.argv) == 1:
        print("No arguments provided. Running predictions on all fires...")
        run_all_fires(target_points=args.points, model_file=args.model)
        sys.exit(0)

    if args.all:
        run_all_fires(target_points=args.points, model_file=args.model)
        sys.exit(0)

    # Validate fire and date for single fire mode
    if not args.fire or not args.date:
        print("Error: Must provide both --fire and --date for single fire mode")
        print("Usage:")
        print("  python3.10 runProduction.py                                    # Run on all fires")
        print("  python3.10 runProduction.py --all                             # Run on all fires")
        print("  python3.10 runProduction.py --fire <fire_name> --date <MMDD>  # Run on specific fire/date")
        sys.exit(1)

    fires = [f for f in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', f)) and not f.startswith('.')]
    if args.fire not in fires:
        print(f"Error: Fire '{args.fire}' not found in training_data folder.")
        sys.exit(1)
    available_dates = rawdata.Day.allGoodDays(args.fire, inference=True)
    if args.date not in available_dates:
        print(f"Error: Date '{args.date}' not available for fire '{args.fire}'. Available dates: {available_dates}")
        sys.exit(1)

    run_production_inference(args.fire, args.date, args.points, args.model) 