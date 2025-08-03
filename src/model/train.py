from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, train_generator, val_generator, config):
    model.compile(
        optimizer=config["training"]["optimizer"],
        loss=config["training"]["loss_function"],
        metrics=config["training"]["metrics"]
    )

    checkpoint = ModelCheckpoint(
        filepath=config["paths"]["model_save_path"],
        monitor=config["callbacks"]["model_checkpoint"]["monitor"],
        save_best_only=config["callbacks"]["model_checkpoint"]["save_best_only"],
        mode=config["callbacks"]["model_checkpoint"]["mode"],
        verbose=config["callbacks"]["model_checkpoint"]["verbose"]
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config["training"]["epochs"],
        callbacks=[checkpoint]
    )

    return model, history
