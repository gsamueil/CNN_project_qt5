import yaml
from Data.dataset_loader import get_data_generators
from model.architecture import build_custom_cnn
from model.train import train_model
from model.evaluate import evaluate_model

if __name__ == "__main__":
    # Load configuration
    with open("src/config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Data
    train_gen, val_gen = get_data_generators(config)
    class_labels = list(train_gen.class_indices.keys())

    # Model
    model = build_custom_cnn(
        input_shape=tuple(config["model"]["input_shape"]),
        num_classes=config["model"]["num_classes"]
    )

    # Train
    model, history = train_model(model, train_gen, val_gen, config)

    # Evaluate
    evaluate_model(
        model,
        val_gen,
        class_labels,
        save_path=config["paths"]["output_metrics"]
    )
