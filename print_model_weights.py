import argparse
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Keras model summary and weights.")
    parser.add_argument('--model', type=str, required=True, help='Path to the .h5 model file')
    parser.add_argument('--num_weather', type=int, default=8, help='Number of weather input features (default: 8)')
    parser.add_argument('--image_feature_size', type=int, default=128, help='Number of image features after flattening (default: 128)')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    print("\nModel Summary:\n")
    model.summary()

    dense_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if weights and weights[0].shape[1] == 1:
                dense_layer = layer
                break
    if dense_layer is None:
        print("Could not find the first dense layer after the merge.")
        exit(1)

    weights, biases = dense_layer.get_weights()
    print(f"\nFirst Dense Layer after merge: {dense_layer.name}")
    print(f"Weights shape: {weights.shape} (input_dim, output_dim)")
    print(f"Biases shape: {biases.shape}")

    #weather weights
    weather_weights = weights[:args.num_weather, 0]
    print("\nWeather input weights (to first dense layer):")
    for i, w in enumerate(weather_weights):
        print(f"  Weather feature {i}: {w}")
    top_weather = np.argsort(np.abs(weather_weights))[::-1][:5]
    print("Top 5 weather weights (by abs value):")
    for idx in top_weather:
        print(f"  Feature {idx}: {weather_weights[idx]}")

    # Image/terrain/perimeter weights: next image_feature_size rows
    image_weights = weights[args.num_weather:, 0]
    print("\nImage/Terrain/Perimeter input weights (to first dense layer):")
    for i, w in enumerate(image_weights):
        print(f"  Image feature {i}: {w}")
    top_image = np.argsort(np.abs(image_weights))[::-1][:5]
    print("Top 5 image/terrain/perimeter weights (by abs value):")
    for idx in top_image:
        print(f"  Feature {idx}: {image_weights[idx]}") 