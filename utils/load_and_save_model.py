import os


def load_model(model_type, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        model = model_type.load_from_checkpoint(checkpoint_path)  # Automatically loads the model with the saved hyperparameters
    else:
        print("Model at location ", checkpoint_path, " not found!")
        exit(1)
    return model

