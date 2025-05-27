
# -- Model V0 : START --
def run_model_v0():
    model_0 = ModelV0(
        input_dimension=360 * 363 * 3, hidden_layer_units=10, output_dimension=8
    ).to(device)

    # Hyperparameters
    epochs = 3
    num_labels = 8
    learning_rate = 0.001

    # Initialize loss function, optimizer, and accuracy metric
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=learning_rate)
    accuracy_metric = Accuracy(
        task="multiclass", num_classes=num_labels, average="macro"
    ).to(device)

    # `train_loader`, `val_loader`, `test_loader` are DataLoader objects
    input_dim = 784  # Example input dimension
    hidden_units = 128  # Example hidden units
    output_dim = 10  # Example output dimension (e.g., 10 classes for MNIST)

    # -- Model V0 : END --

    # train_and_test_from_youtube(model=model_0, train_dataloader=train_loader, test_dataloader=test_loader,
    #                             loss_fn=loss_function, optimizer=optimizer, accuracy_fn=accuracy_fn, epochs=epochs, device=device)

    # train_and_evaluateClaude(
    #     epochs=epochs,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     model=model_0,
    #     loss_function=loss_function,
    #     optimizer=optimizer,
    #     accuracy_metric=accuracy_metric,
    #     device=device,
    #     num_classes=num_labels,
    #     class_names=class_names,
    #     debug=False,
    # )
    train_model(epochs, train_loader, val_loader, model_0, loss_function,
                optimizer, accuracy_metric, device, num_classes=num_labels, debug=False)

    test_model(test_loader, model_0, loss_function, accuracy_metric, device, num_classes=num_labels)
    # model_0 = train_validate_test_with_torchmetrics(
    #     model=model_0,
    #     train_loader=train_loader,
    #     num_classes=8,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     epochs=10,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=device,
    # )

    # Load the model results
    # loaded_history = load_model_results(model_0, 'results/model_v0.pth')
    model_details = {
        "model_name": model_0.__class__.__name__,
        "learning_rate": learning_rate.__str__(),
        "loss_function": loss_function.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "accuracy_metric": accuracy_metric.__class__.__name__,
        "epochs": epochs.__str__(),
    }
    plot_model_performance(model_0, class_names, model_details=model_details)
    print("Model training and evaluation completed successfully.")

# Model V1 : START
def run_model_v1():
    input_dimension = 360 * 363 * 3

    model_1 = ModelV1(
        input_dimension=input_dimension,
        hidden_layer_units=10,
        output_dimension=8,
    ).to(device)

    # Stages:
    # For [3 hidden layers with 64, 32, 10 units]:
    # 1. run this model and show its bad performance. Show the confusion matrix and how it only predicts one class. (lr=0.0001, epochs=5, optimizer = optim.Adam).
    # 2. run this model with other configurations and show there is no improvement in the confusion matrix. (lr=0.001, epochs=5, optimizer = torch.optim.SGD)
    # 3. change the model to have 3 hidden layers with 64, 32, 10 units. Show the improvement in the confusion matrix. (lr=0.001, epochs=5, optimizer = torch.optim.SGD)

    # Hyperparameters
    epochs = 5
    num_labels = 8
    learning_rate = 0.1

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_1.parameters(), lr=learning_rate)
    accuracy_metric = Accuracy(
        task="multiclass", num_classes=num_labels, average="macro"
    ).to(device)

    train_model(epochs, train_loader, val_loader, model_1, loss_function,
                optimizer, accuracy_metric, device, num_classes=num_labels, debug=False)

    test_model(test_loader, model_1, loss_function, accuracy_metric, device, num_classes=num_labels)

    model_details = {
        "model_name": model_1.__class__.__name__,
        "learning_rate": learning_rate.__str__(),
        "loss_function": loss_function.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "accuracy_metric": accuracy_metric.__class__.__name__,
        "epochs": epochs.__str__(),
    }

    plot_model_performance(model_1, class_names, model_details=model_details)

    print("Model training and evaluation completed successfully.")
# --- Model V1 : END ---
