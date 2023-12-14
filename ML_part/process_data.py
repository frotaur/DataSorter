import os
import pickle
import torch

def format_dict_for_py(data_dict):
    formatted_dict = {}
    for key, value in data_dict.items():
        formatted_value = {}
        for sub_key, sub_value in value.items():
            if sub_key != "k_size":  # Skip "k_size" in nested dictionaries
                formatted_value[sub_key] =  sub_value
        formatted_dict[key] = formatted_value
    return formatted_dict

def update_and_process_data(input_directory, output_py_file, output_pth_file, target_device='cuda:0'):
    """
    Update and process data from pickle files in the input directory.

    Args:
    input_directory: Directory containing pickle files with data.
    output_py_file: Path to the output Python file.
    output_pth_file: Path to the output PyTorch tensor file.
    target_device: Device for PyTorch tensors (e.g., "cuda:0" for GPU or "cpu" for CPU).

    Description:
    This function updates existing data with new data from pickle files in the input directory. It loads existing data
    if available, updates it with new data, and processes the data for further use.

    The updated data is formatted for Python file output, removing the "k_size" key from nested dictionaries. The
    formatted data is saved to a Python file.

    Additionally, the function processes the data into PyTorch tensors, ensuring that each parameter value is a tensor.
    The processed data is saved to a PyTorch tensor file.
    """
    def load_pickle_files(directory, existing_data):
        for filename in os.listdir(directory):
            if filename.endswith(".pk"):
                key = os.path.splitext(filename)[0][:10]
                # Only add new data
                if key not in existing_data:
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                        existing_data[key] = content
        return existing_data

    # Load existing data if it exists
    existing_data = {}
    if os.path.exists(output_py_file):
        with open(output_py_file, 'r') as f:
            exec(f.read())
            existing_data = dictionary_data

    # Update existing data with new data from the input directory
    updated_data = load_pickle_files(input_directory, existing_data)

    dict_data_one_params = {}
    for key, value in updated_data.items():
        if "params_a" in value and "params_d" in value and "t_crit" in value:
            params_a = value['params_a']
            params_d = value['params_d']
            t_crit = value['t_crit']

            new_params = {}
            for param_key in params_a.keys():
                if param_key != "k_size":  # Skip "k_size"
                    new_params[param_key] = params_d[param_key] *  (1 - t_crit) + params_a[param_key] * t_crit

            dict_data_one_params[key] = {'params': new_params}

    # Format the dictionary for Python file
    formatted_data = format_dict_for_py(dict_data_one_params)

    # Save the formatted dictionary to a .py file
    with open(output_py_file, 'w') as f:
        f.write("from torch import tensor\n\n")
        f.write("dictionary_data = ")
        f.write(repr(formatted_data))

    print(f"Updated dictionary saved to {output_py_file}")
    print(f"Size of updated dictionary: {len(formatted_data)}")

    dict_data_tensors = {}
    for key, value in dict_data_one_params.items():
        params = value['params']

        tensors_list = []
        for param_key, param_value in params.items():
            # Ensure param_value is a tensor
            if not isinstance(param_value, torch.Tensor):
                param_value = torch.tensor([param_value], device=target_device)
            else:
                param_value = param_value.to(target_device)
            tensors_list.append(param_value.flatten())

        concatenated_tensor = torch.cat(tensors_list)
        dict_data_tensors[key] = concatenated_tensor

    print(dict_data_tensors)
    torch.save(dict_data_tensors, output_pth_file)
    print(f"Processed data saved to {output_pth_file}")

if __name__=='__main__':
    # Set up
    input_directory = "/home/Zilan/Desktop/leniasearch/ML_part/Hashed"
    output_py_file = "/home/Zilan/Desktop/leniasearch/ML_part/dictionary_data.py"
    output_pth_file = "/home/Zilan/Desktop/leniasearch/ML_part/dict_data_tensors.pth"

    # Run the function
    update_and_process_data(input_directory, output_py_file, output_pth_file)
