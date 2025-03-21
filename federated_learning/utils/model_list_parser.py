def get_worker_num_from_model_file_name(model_file_name):
    """
    :param model_file_name: string
    """
    return int(model_file_name.split("_")[1])

def get_epoch_num_from_model_file_name(model_file_name):
     """
     :param model_file_name: string
     """
     return int(model_file_name.split("_")[2].split(".")[0])

# def get_epoch_num_from_model_file_name(model_file_name):
#     parts = model_file_name.split("_")
    
#     # Check if the filename has at least 3 parts
#     if len(parts) >= 3:
#         epoch_part = parts[2].split(".")[0]
        
#         if epoch_part.isdigit():  # Ensure it's numeric before converting
#             return int(epoch_part)
#         else:
#             raise ValueError(f"Expected a numeric epoch but got '{epoch_part}' in file: {model_file_name}")
#     else:
#         # Raise a more informative error if the file name format is wrong
#         raise ValueError(f"Unexpected model file name format: {model_file_name}")


def get_suffix_from_model_file_name(model_file_name):
    """
    :param model_file_name: string
    """
    return model_file_name.split("_")[3].split(".")[0]

def get_model_files_for_worker(model_files, worker_id):
    """
    :param model_files: list[string]
    :param worker_id: int
    """
    worker_model_files = []

    for model in model_files:
        worker_num = get_worker_num_from_model_file_name(model)

        if worker_num == worker_id:
            worker_model_files.append(model)

    return worker_model_files

def get_model_files_for_epoch(model_files, epoch_num):
    """
    :param model_files: list[string]
    :param epoch_num: int
    """
    epoch_model_files = []

    for model in model_files:
        model_epoch_num = get_epoch_num_from_model_file_name(model)

        if model_epoch_num == epoch_num:
            epoch_model_files.append(model)

    return epoch_model_files

def get_model_files_for_suffix(model_files, suffix):
    """
    :param model_files: list[string]
    :param suffix: string
    """
    suffix_only_model_files = []

    for model in model_files:
        model_suffix = get_suffix_from_model_file_name(model)

        if model_suffix == suffix:
            suffix_only_model_files.append(model)

    return suffix_only_model_files
