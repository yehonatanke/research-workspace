import shutil # Import the shutil module for file operations

def prepare_yolo_dataset(source_path, destination_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepares a YOLO-compliant dataset structure by splitting the data into training,
    validation, and testing subsets.

    Args:
        source_path (str): Path to the original dataset directory.
        destination_path (str): Path for the new YOLO-compliant dataset structure.
        train_ratio (float): Proportion of data for training (default: 0.7).
        val_ratio (float): Proportion of data for validation (default: 0.15).
        test_ratio (float): Proportion of data for testing (default: 0.15).
    """
    # Validate split ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Create YOLO directory structure
    folders = ['train', 'val', 'test']
    for folder in folders:
        for blood_cell_type in os.listdir(source_path):
            cell_path = os.path.join(source_path, blood_cell_type)
            if os.path.isdir(cell_path):
                os.makedirs(os.path.join(destination_path, folder, blood_cell_type), exist_ok=True)

    # Process each blood cell type
    for blood_cell_type in os.listdir(source_path):
        cell_path = os.path.join(source_path, blood_cell_type)
        if not os.path.isdir(cell_path):
            continue

        # Get all image files
        images = [f for f in os.listdir(cell_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Create labels for stratification
        labels = [blood_cell_type] * len(images)

        # Split into train, test, and val using stratify
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            images, labels, test_size=(test_ratio + val_ratio), stratify=labels, random_state=42
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=test_ratio / (test_ratio + val_ratio), stratify=temp_labels, random_state=42
        )

        # Copy images to the respective folders
        for image in train_images:
            shutil.copy(os.path.join(cell_path, image), os.path.join(destination_path, 'train', blood_cell_type, image))

        for image in val_images:
            shutil.copy(os.path.join(cell_path, image), os.path.join(destination_path, 'val', blood_cell_type, image))

        for image in test_images:
            shutil.copy(os.path.join(cell_path, image), os.path.join(destination_path, 'test', blood_cell_type, image))

    # Verify the dataset structure
    print("--- Dataset Summary ---")
    for folder in folders:
        for blood_cell_type in os.listdir(source_path):
            check_path = os.path.join(destination_path, folder, blood_cell_type)
            if os.path.exists(check_path):
                print(f"{folder}/{blood_cell_type}: {len(os.listdir(check_path))} images")

    print("--- Dataset preparation is complete ---")


# Define a function to train YOLO model with your DataLoader
def train_yolo_with_dataloader(dataloader, classes=8, epochs=50):
    # Initialize YOLO model for classification
    # Using YOLOv8n-cls model which is designed for classification tasks
    model = YOLO('yolov8n-cls.pt')

    # Set model parameters
    model.overrides['task'] = 'classify'  # Set task to classification
    model.overrides['imgsz'] = 224  # Match your image size
    model.overrides['classes'] = classes  # Set number of classes

    # Train the model
    results = model.train(
        data=None,  # We'll use our custom DataLoader instead
        epochs=epochs,
        batch=dataloader.batch_size,
        imgsz=224,
        device=0 if torch.cuda.is_available() else 'cpu',
        exist_ok=True
    )

    return model, results

# Define a function to evaluate the model with your DataLoader
def evaluate_yolo_with_dataloader(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)

            # Get predictions
            results = model(images)
            probs = results[0].probs  # Get probability distributions
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Validation Accuracy: {accuracy:.4f}")

    return accuracy, all_preds, all_labels
