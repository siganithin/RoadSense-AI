import kagglehub

# Download the dataset
print("Downloading dataset... This may take a few minutes.")
path = kagglehub.dataset_download("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")
print(f"Dataset downloaded to: {path}")