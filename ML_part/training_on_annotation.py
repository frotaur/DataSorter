import torch
import json
import os
from .model import EnsemblePredictor, SymmetricDNN

def get_model_architecture(model_path):
    model_state = torch.load(model_path)
    state_dict = model_state['state_dict']
    architecture = {}
    for param_tensor in state_dict:
        architecture[param_tensor] = state_dict[param_tensor].size()
    return architecture


def train_on_annotation(json_path='./annotations_data.json', 
                        tensor_data_path='./dict_data_tensors.pth', 
                        from_scratch=False, 
                        save_updated_models=True, 
                        base_model_class = SymmetricDNN, 
                        num_predictors = 5, 
                        hidden_layers =[1000, 6], 
                        device ="cuda:0", 
                        saved_models_dir='./saved_models', 
                        epochs=10, 
                        batch_size=20, 
                        with_bootstrap = True):
    """        
    Train an ensemble of models for annotation data.
    /!\\/!\\/!\\ Use one directory per EnsembleOfPredictors to save and load the model /!\\/!\\/!\\

    Args:
    json_path: Path to the JSON file containing annotations data.
    tensor_data_path: Path to the preprocessed parameters in tensor format.
    from_scratch: Whether to start training from scratch or continue from saved models.
    save_updated_models: Whether to save the updated models after training.
    base_model_class: Class of the base model to be used in the ensemble.
    num_predictors: Number of predictors in the ensemble.
    hidden_layers: List specifying the architecture of the base model's hidden layers.
    device: Device for training (e.g., "cuda:0" for GPU or "cpu" for CPU).
    saved_models_dir: Directory to save/load trained models.
    epochs: Number of training epochs.
    batch_size: Batch size for training.
    with_bootstrap: Whether to use bootstrap sampling during training.

    Description:
    This script loads preprocessed data and annotations from specified paths, prepares inputs, and trains an ensemble
    of models using the specified configuration. It allows for continuing training from saved models or starting
    training from scratch. The trained models can be saved if desired.
    """

    # Load preprocessed data
    dict_data_tensors = torch.load(tensor_data_path)

    # Load annotations data from JSON
    with open(json_path, 'r') as file:
        score_data = json.load(file)

    inputs_left = []
    inputs_right = []
    outputs = []

    for matchup in score_data:
        left_params = dict_data_tensors.get(matchup['left'])
        right_params = dict_data_tensors.get(matchup['right'])
        
        if left_params is not None and right_params is not None:
            inputs_left.append(left_params)
            inputs_right.append(right_params)
            outputs.append(matchup['side'])

    # Inputs creation
    inputs_left = torch.stack(inputs_left)
    inputs_right = torch.stack(inputs_right)
    outputs = torch.tensor(outputs, dtype=torch.float32)[:, None]  # Make it (B,1) format

    # Determine model architecture from saved models or default settings
    if not from_scratch and saved_models_dir is not None and os.path.exists(saved_models_dir):
        example_model_path = os.path.join(saved_models_dir, 'predictor_0.pt')
        if os.path.exists(example_model_path):
            model_architecture = get_model_architecture(example_model_path)
            old_input_dim = model_architecture['model.0.weight'][1]
            old_hidden_layers = [model_architecture['model.0.weight'][0], model_architecture['model.2.weight'][0]]
            ensemble = EnsemblePredictor(base_model_class, num_predictors=num_predictors, input_dim=old_input_dim, hidden_layers=old_hidden_layers, device=device)
            ensemble.load_models(saved_models_dir)  # Load pre-trained models
        else:
            raise FileNotFoundError("Saved model not found")
    else:
        input_dim = 2*inputs_left.size(1)
        ensemble = EnsemblePredictor(base_model_class, num_predictors=num_predictors, input_dim=input_dim, hidden_layers=hidden_layers, device=device)

    # Train the ensemble
    ensemble.train(x=inputs_left, y=inputs_right, true_label=outputs, epochs=epochs, batch_size=batch_size, with_bootstrap = with_bootstrap)

    # Save updated models
    if save_updated_models:
        ensemble.save_models(saved_models_dir)

if __name__=='__main__':
    # Set up and train
    train_on_annotation(json_path='./annotations_data.json',
                        tensor_data_path='./dict_data_tensors.pth',
                        base_model_class=SymmetricDNN,
                        num_predictors=3,
                        hidden_layers=[1000, 6],
                        device="cuda:0",
                        saved_models_dir='./saved_models',
                        from_scratch=True,
                        epochs=100,
                        batch_size=20,
                        save_updated_models=True, 
                        with_bootstrap = True)