from model_functions import *


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset
    print("Generating dataset...")
    fully_sampled, undersampled = generate_dataset(
        num_samples=1000,         # Generate 100 brain structures
        volume_shape=(64, 64, 64),
        slices_per_sample=3,     # Get 3 slices per sample
        undersample_factor=4,    # Keep 1 out of every 4 lines
        central_lines_zero_percent=0.05  # Zero out 5% of central lines
    )

    print(f"Generated {len(fully_sampled)} pairs of slices")

    # Normalize and reshape data
    fully_sampled = (fully_sampled - np.min(fully_sampled)) / \
        (np.max(fully_sampled) - np.min(fully_sampled))
    undersampled = (undersampled - np.min(undersampled)) / \
        (np.max(undersampled) - np.min(undersampled))

    # Add channel dimension if needed
    if fully_sampled.ndim == 3:
        fully_sampled = fully_sampled[:, :, :, np.newaxis]
        undersampled = undersampled[:, :, :, np.newaxis]

    # Move channels dimension to pytorch format: [batch, channels, height, width]
    fully_sampled = np.transpose(fully_sampled, (0, 3, 1, 2))
    undersampled = np.transpose(undersampled, (0, 3, 1, 2))

    # Split the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        undersampled, fully_sampled, test_size=0.3, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    print("Data split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Create datasets
    train_dataset = MRIDataset(X_train, y_train)
    val_dataset = MRIDataset(X_val, y_val)
    test_dataset = MRIDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create all three models
    models = {
        "spatial": SpatialDomainUNet(in_channels=1, out_channels=1, base_filters=64),
        "frequency": FrequencyDomainUNet(in_channels=1, out_channels=1, base_filters=64),
        "dual": DualDomainUNet(in_channels=1, out_channels=1, base_filters=32),
        "SequentialHybrid": SequentialHybridModel2(in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=False),
        "SequentialHybrid_Skips": SequentialHybridModel2(in_channels=1, out_channels=1, base_filters=64, depth=4, use_fusion_skips=True)
    }

    # Train and evaluate each model
    histories = {}
    metrics_results = {}
    test_outputs = {}

    # Loss function
    criterion = nn.MSELoss()

    # Train each model
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} domain model ---\n")
        if "Sequential" in model_name:
            combined_loss = True
        else:
            combined_loss = False
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Train model
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=50, patience=10, model_name=f"mri_{model_name}_model", combined_loss=combined_loss
        )

        # Save history
        histories[model_name] = history

        # Plot training history
        plot_history(
            history, title=f"{model_name.capitalize()} Domain Model Training")

        # Evaluate on test set
        print(f"\nEvaluating {model_name} domain model...")
        metrics, test_input, test_output, test_target = compute_metrics(
            trained_model, test_loader, device)

        print(f"\n{model_name.capitalize()} Domain Model Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics and test outputs
        metrics_results[model_name] = metrics
        test_outputs[model_name] = test_output

        # Visualize results
        visualize_results(test_input, test_output, test_target, num_samples=3,
                          title=f"{model_name.capitalize()} Domain Model Results")

    # # Compare models
    compare_models(
        [metrics_results["spatial"], metrics_results["frequency"], metrics_results["dual"],
            metrics_results["SequentialHybrid"], metrics_results["SequentialHybrid_Skips"]],
        ["Spatial", "Frequency", "Dual", "SequentialHybrid", "SequentialHybrid_Skips"]
    )


if __name__ == '__main__':
    main()
