#!/usr/bin/env python3.10

import sys
import os
import random
import argparse
import matplotlib.pyplot as plt
import cv2
from lib import rawdata
from lib import dataset
from lib import model
from lib import viz
from lib import preprocess
import numpy as np

def list_fires_and_dates():
    fires = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f)) and not f.startswith('.')]
    print('Available fires:')
    for fire in fires: 
        try:
            dates = rawdata.Day.allGoodDays(fire)
            print(f"  {fire}: {dates}")
        except Exception as e:
            print(f"  {fire}: Error reading dates ({e})")

def run_fire_with_filename(fire_name, date, target_points=10000, eval_mode=True):
    print(f"Loading fire data for {fire_name} on date {date}...")
    
    # Load data for the specified fire and date
    data = rawdata.RawData.load(burnNames=[fire_name], dates={fire_name: [date]}, inference=not eval_mode)
    
    # Create dataset with vulnerable pixels around fire perimeter
    test_dataset = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    
    print(f"Created dataset with {len(test_dataset)} total points")
    
    # Sample points to reduce density while maintaining coverage
    all_points = test_dataset.toList(test_dataset.points)
    if len(all_points) > target_points:
        print(f"Sampling {target_points} points from {len(all_points)} total points (balanced sampling)")
        sampled_points = random.sample(all_points, target_points)
        test_dataset = dataset.Dataset(data, sampled_points)
    
    # Show available dates
    for burnName, date in test_dataset.getUsedBurnNamesAndDates():
        points = test_dataset.points[burnName][date]
        print(f"{burnName} {date}: {len(points)} points")
    
    # Load the most recent model
    model_file = "20200903-193223mod"
    print(f"Loading model: {model_file}")
    
    # Get model and preprocessor
    mod, pp = getModel(model_file)
    
    # Preprocess (returns only inputs and ptList in inference mode)
    inputs, ptList = pp.process(test_dataset, inference=not eval_mode)
    # Robustly flatten and take only the first two arrays for model input
    flat_inputs = flatten_inputs(inputs)
    model_inputs = flat_inputs[:2]
    print("inputs type:", type(model_inputs))
    for i, arr in enumerate(model_inputs):
        print(f"Input {i}: shape={np.shape(arr)}, dtype={getattr(arr, 'dtype', type(arr))}")
    # Predict
    predictions = mod.predict(model_inputs)
    
    print(f"Generated {len(predictions)} predictions")
    
    # Calculate performance and visualize only if eval_mode
    if eval_mode:
        print("Calculating performance and generating visualizations...")
        calculatePerformance(test_dataset, predictions, target_points, ptList)
    else:
        print("Skipping performance metrics: running in inference/production mode (no ground truth available)")
    
    return test_dataset, predictions

def getModel(weightsFile=None):
    print('in getModel')
    numWeatherInputs = 8
    usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)
    mod = model.fireCastModel(pp, weightsFile)
    return mod, pp

def calculatePerformance(test, predictions, point_count, ptList):
    fireDate = []
    samples = []
    preResu = []

    print("SIZE OF PREDICTIONS: ", len(predictions))
    for pt, pred in zip(ptList, predictions):
        fireDate.append(pt[1])
        samples.append(pt[2])
        preResu.append(pred)

    viz.getNumbers(test, samples, preResu, len(predictions), fireDate)
    res = viz.visualizePredictions(test, dict(zip(ptList, predictions)), preResu)
    savePredictionsWithFilename(res, point_count)

def savePredictionsWithFilename(predictionsRenders, point_count):
    radius = dataset.Dataset.VULNERABLE_RADIUS
    burns = {}
    for (burnName, date), render in predictionsRenders.items():
        if burnName not in burns:
            burns[burnName] = []
        burns[burnName].append((date, render))
    for burnName, frameList in burns.items():
        frameList.sort()
        fig = plt.figure(burnName, figsize=(8, 6))
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for date, render in frameList:
            withTitle = render.copy()
            cv2.putText(withTitle,date, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(withTitle)
            fname = f"output/figures/{burnName}_{date}_radius{radius}_points{point_count}.png"
            plt.savefig(fname,bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved visualization to: {fname}")

def flatten_inputs(inputs):
    flat = []
    if isinstance(inputs, (tuple, list)):
        for item in inputs:
            if isinstance(item, (tuple, list)):
                flat.extend(flatten_inputs(item))
            else:
                flat.append(item)
    else:
        flat.append(inputs)
    return flat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fire prediction with probability-based coloring for any fire/date')
    parser.add_argument('--fire', type=str, help='Fire name (as in data folder)')
    parser.add_argument('--date', type=str, help='Date in MMDD format')
    parser.add_argument('--points', type=int, default=10000, help='Number of points to sample (default: 10000)')
    parser.add_argument('--list', action='store_true', help='List available fires and dates')
    parser.add_argument('--eval', dest='eval_mode', action='store_true', help='Run in evaluation mode (requires next day perim, computes metrics)')
    parser.add_argument('--no-eval', dest='eval_mode', action='store_false', help='Run in inference/production mode (no metrics, no next day perim required)')
    parser.set_defaults(eval_mode=True)
    args = parser.parse_args()

    if args.list or (not args.fire or not args.date):
        list_fires_and_dates()
        print("\nTo run: python3.10 runPrediction.py --fire <fire_name> --date <MMDD> [--points N] [--eval/--no-eval]")
        sys.exit(0)

    # Validate fire and date
    fires = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f)) and not f.startswith('.')]
    if args.fire not in fires:
        print(f"Error: Fire '{args.fire}' not found in data folder. Use --list to see available fires.")
        sys.exit(1)
    available_dates = rawdata.Day.allGoodDays(args.fire, inference=not args.eval_mode)
    if args.date not in available_dates:
        print(f"Error: Date '{args.date}' not available for fire '{args.fire}'. Available dates: {available_dates}")
        sys.exit(1)

    run_fire_with_filename(args.fire, args.date, args.points, eval_mode=args.eval_mode) 