### Imports ###

import os
import torch
import torch.nn as nn
import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score


### Bootstrap sample ###

def bootstrap_sample(x, y, result, size):
    indices = np.random.choice(len(x), size=size, replace=True)
    x_sample = x[indices]
    y_sample = y[indices]
    result_sample = result[indices]

    # Identify the out-of-bag (OOB) indices and select the OOB sample
    oob_indices = np.setdiff1d(np.arange(len(x)), indices)
    x_oob = x[oob_indices]
    y_oob = y[oob_indices]
    result_oob = result[oob_indices]
    
    return x_sample, y_sample, result_sample, x_oob, y_oob, result_oob


### Definition of the class model ###

class SymmetricDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], init_reg_coeff = 0.2, reg_coeff_adaptation = 1.02, device='cpu'):
        super(SymmetricDNN, self).__init__()
        self.input_dim = input_dim  # This should be 2*k, where k is the number of parameters of a simulation
        self.hidden_layers = hidden_layers
        self.model = self.build_model()
        self.reg_coeff = init_reg_coeff
        self.reg_coeff_adaptation = reg_coeff_adaptation

        self.device = device
        self.to(device)

    def get_size(self):
        return self.hidden_layers

    def build_model(self) -> nn.Sequential:
        layers = []
        input_size = self.input_dim
        
        # Add hidden layers
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # Using ReLU activation function
            input_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())  # Using Sigmoid activation function for binary classification
        
        return nn.Sequential(*layers)
    

    def symmetricLoss(self,pred_xy: torch.Tensor, pred_yx: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
            WARNING DO NOT USE !!! IT BREAKS DOWN WHEN LOG(0), use sym_loss_pytorch instead
            Args :
            pred_xy/yx : (B, 1) tensor of probabilities, s.t. pred_xy = 1 - pred_yx
            true_labels : (B, 1) tensor of binary labels, either 0 or 1
        """
        
        raise NotImplementedError("This function is broken, use sym_loss_pytorch instead")
        # Loss for DNN(x, y) and 1 - DNN(y, x)
        loss_xy = - (true_labels * torch.log(pred_xy) + (1 - true_labels) * torch.log(1 - pred_xy))
        loss_yx = - ((1 - true_labels) * torch.log(pred_yx) + true_labels * torch.log(1 - pred_yx))
        
        # Combined symmetric loss
        loss = loss_xy + loss_yx

        #print("loss xy : ", torch.mean(loss_xy), "loss yx : ", torch.mean(loss_yx))
        #print(pred_xy, pred_yx, true_labels)
        return torch.mean(loss)
    
    def sym_loss_pytorch(self,pred_xy: torch.Tensor,pred_yx: torch.Tensor,  true_labels: torch.Tensor) -> torch.Tensor:
        """
            Symmetric binary cross-entropy loss function.
        """
        loss_xy = torch.nn.BCELoss(reduction='mean')(pred_xy, true_labels)
        loss_yx = torch.nn.BCELoss(reduction='mean')(pred_yx, 1 - true_labels)

        return loss_xy + loss_yx

    def compute_symmetric_output(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        xy = np.concatenate((x, y), axis=1)
        yx = np.concatenate((y, x), axis=1)
        #print("xy = ", xy)
        
        xy_tensor = torch.tensor(xy, dtype=torch.float32)
        yx_tensor = torch.tensor(yx, dtype=torch.float32)
        
        out_xy = self.model(xy_tensor).detach().numpy()
        out_yx = self.model(yx_tensor).detach().numpy()
        
        return out_xy, out_yx
    
    def train_model_without_bootstrap(self, x, y, true_label, epochs: int, batch_size: int):
        """
            Train the model without bootstrap sampling.

            Args:
            x/y: (B, D) batch of x/y parameters
            true_label: (B, 1) batch of binary labels
            epochs: number of epochs
            batch_size: batch size
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
        xy_tensor = torch.cat((x, y), dim=1).to(dtype=torch.float32, device=self.device) # (B, 2D)
        yx_tensor = torch.cat((y, x), dim=1).to(dtype=torch.float32, device=self.device) # (B, 2D)
        true_label_tensor = torch.tensor(true_label, dtype=torch.float32, device=self.device) # (B, 1)
        
        assert true_label_tensor.shape[0] == xy_tensor.shape[0]

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            xy_outputs = self.model(xy_tensor)
            yx_outputs = self.model(yx_tensor)
    
            # Compute Loss
            loss = self.sym_loss_pytorch(xy_outputs, yx_outputs, true_label_tensor)
            
            # Backward pass
            if loss > 1e-7:
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')


    def train_model_with_bootstrap(self, x, y, true_label, epochs, batch_size):
        """
        Train the model with bootstrap sampling, allowing for regularization adjustment during training.

        Args:
        x/y: (B, D) batch of x/y parameters
        true_label: (B, 1) batch of binary labels
        epochs: number of epochs
        batch_size: batch size

        Description:
        This function trains the model using bootstrap sampling, which involves repeatedly selecting random subsets of the
        input data (x, y, true_label) with replacement for each training iteration. It also allows for dynamic adjustment
        of the regularization coefficient (reg_coeff) based on the ratio of validation loss to training loss.

        The optimization process is as follows:
        - The optimizer is defined with an initial regularization coefficient (reg_coeff) and a fixed learning rate.
        - Training and validation datasets are created using bootstrap_sample() function.
        - For each epoch, the model is trained on the training dataset, and the loss is computed for both training and
          validation datasets.
        - If the training loss is above a small threshold (1e-7), the backward pass is performed, and the optimizer updates
          the model's parameters.
        - The function monitors the ratio of validation loss to training loss. If the ratio is less than 1.1, it decreases
          the regularization coefficient to combat overfitting; if it's greater than 1.5, it increases the coefficient to
          address underfitting.
        - The optimizer's weight_decay parameter is updated with the new_reg_coeff to apply regularization accordingly.

        Additionally, the function prints the training loss, validation loss, and the current regularization coefficient at
        regular intervals to track the model's progress during training.
        """

        # Define the optimizer with the initial regularization coefficient
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=self.reg_coeff)

        # Get the training and the testing data
        x_train, y_train, true_label_train, x_val, y_val, true_label_val = bootstrap_sample(x, y, true_label, batch_size)

        xy_train_tensor = torch.cat((x_train, y_train), dim=1).to(dtype=torch.float32, device=self.device)
        yx_train_tensor = torch.cat((y_train, x_train), dim=1).to(dtype=torch.float32, device=self.device)
        true_label_train_tensor = torch.tensor(true_label_train).to(dtype=torch.float32, device=self.device)

        xy_val_tensor = torch.cat((x_val, y_val), dim=1).to(dtype=torch.float32, device=self.device)
        yx_val_tensor = torch.cat((y_val, x_val), dim=1).to(dtype=torch.float32, device=self.device)
        true_label_val_tensor = torch.tensor(true_label_val).to(dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            xy_outputs_train = self.model(xy_train_tensor)
            yx_outputs_train = self.model(yx_train_tensor)
            xy_outputs_val = self.model(xy_val_tensor)
            yx_outputs_val = self.model(yx_val_tensor)

            # Compute Training Loss
            loss_train = self.sym_loss_pytorch(xy_outputs_train, yx_outputs_train, true_label_train_tensor)

            # Compute Validation Loss
            loss_val = self.sym_loss_pytorch(xy_outputs_val, yx_outputs_val, true_label_val_tensor)

            #print("Loss train = ", loss_train)
            # Backward pass and optimize
            if loss_train > 1e-7:
                
                loss_train.backward()
                optimizer.step()

                loss_ratio = loss_val.item() / loss_train.item()

                # Adjust reg_coeff based on the ratio of val_loss and train_loss
                new_reg_coeff = self.reg_coeff
                if loss_ratio < 1.1:
                    new_reg_coeff /= self.reg_coeff_adaptation  # decrease regularization if overfitting
                elif loss_ratio > 1.5:
                    new_reg_coeff *= self.reg_coeff_adaptation  # increase regularization if underfitting

                # Update optimizer with new_reg_coeff
                if new_reg_coeff != self.reg_coeff:
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] = new_reg_coeff
                    self.reg_coeff = new_reg_coeff

                

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss_train.item()}, Val Loss: {loss_val.item()}, Reg Coeff: {self.reg_coeff}')

    
    def predict(self, x: np.array, y: np.array) -> np.array:
        self.model.eval()
        
        with torch.no_grad():
            xy_tensor = torch.cat((x, y), dim=1).to(dtype=torch.float32, device=self.device)

            return self.model(xy_tensor).detach()
    
    def binary_predict(self, x: np.array, y: np.array) -> np.array:
        predictions = self.predict(x, y)
        
        # Convert predictions to binary labels
        binary_predictions = (predictions > 0.5).astype(int)
        
        return binary_predictions
    
### Ensemble of Predictors ###

class EnsemblePredictor:
    def __init__(self, base_model_class, num_predictors, *model_args, **model_kwargs):
        """
        Initialize an ensemble predictor with multiple base models.

        Args:
        base_model_class: Class of the base model to be used in the ensemble.
        num_predictors: Number of predictors in the ensemble.
        *model_args: Additional positional arguments for the base model constructor.
        **model_kwargs: Additional keyword arguments for the base model constructor.
        """
        self.predictors = [base_model_class(*model_args, **model_kwargs) for _ in range(num_predictors)]
        self.num_predictors = num_predictors
    
    def train(self, x, y, true_label, epochs, batch_size, with_bootstrap):
        """
        Train the ensemble of predictors.

        Args:
        x/y: (B, D) batch of x/y parameters
        true_label: (B, 1) batch of binary labels
        epochs: number of epochs for training
        batch_size: batch size for training
        with_bootstrap: whether to use bootstrap sampling during training

        Description:
        This method trains each predictor in the ensemble by calling either 'train_model_with_bootstrap' or
        'train_model_without_bootstrap' based on the 'with_bootstrap' flag. It prints progress information during training.
        """
        for i, predictor in enumerate(self.predictors):
            print(f"Training Predictor {i+1}/{len(self.predictors)}")
            if with_bootstrap:
                predictor.train_model_with_bootstrap(x, y, true_label, epochs, batch_size)
            else:
                predictor.train_model_without_bootstrap(x, y, true_label, epochs, batch_size)

    def predict(self, x, y):
        """
        Make predictions using the ensemble of predictors.

        Args:
        x/y: (B, D) batch of x/y parameters

        Returns:
        predictions: Mean predictions across all predictors in the ensemble.

        Description:
        This method makes predictions using each predictor in the ensemble and returns the mean of these predictions.
        It also prints the input parameters and predictions for debugging purposes.
        """
        predictions = [predictor.predict(x, y) for predictor in self.predictors]
        print(x, y)
        print(predictions)
        print("Mean = ", np.mean(predictions, axis=0))
        return np.mean(predictions, axis=0)
    
    def mean_and_variance(self, x, y):
        """
        Calculate the mean and variance of predictions using the ensemble of predictors.

        Args:
        x/y: (B, D) batch of x/y parameters

        Returns:
        mean: Mean of predictions across all predictors in the ensemble.
        var: Variance of predictions across all predictors in the ensemble.

        Description:
        This method computes predictions using each predictor in the ensemble, stacks them into a tensor, and then
        calculates the mean and variance of these predictions.
        """
        predictions = [predictor.predict(x, y) for predictor in self.predictors]
        predictions_tensor = torch.stack(predictions, dim=0)

        mean = torch.mean(predictions_tensor)
        var = torch.var(predictions_tensor)
        
        return mean, var

    def save_models(self, saved_models_dir):
        """
        Save the state of each predictor in the ensemble to separate files.

        Args:
        saved_models_dir: Directory where model states will be saved.

        Description:
        This method saves the state of each predictor in the ensemble, including its model parameters and regularization
        coefficient, to separate files in the specified directory.
        """
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)
        for i, predictor in enumerate(self.predictors):
            model_state = {
                'state_dict': predictor.state_dict(),
                'reg_coeff': predictor.reg_coeff
            }
            torch.save(model_state, os.path.join(saved_models_dir, f'predictor_{i}.pt'))

    def load_models(self, saved_models_dir):
        """
        Load the state of each predictor in the ensemble from separate files.

        Args:
        saved_models_dir: Directory from which model states will be loaded.

        Description:
        This method loads the state of each predictor in the ensemble from separate files in the specified directory,
        restoring their model parameters and regularization coefficients.
        """
        for i, predictor in enumerate(self.predictors):
            model_path = os.path.join(saved_models_dir, f'predictor_{i}.pt')
            if os.path.exists(model_path):
                model_state = torch.load(model_path)
                predictor.load_state_dict(model_state['state_dict'])
                predictor.reg_coeff = model_state['reg_coeff']
            else:
                raise FileNotFoundError(f"Model at {model_path} not found")
