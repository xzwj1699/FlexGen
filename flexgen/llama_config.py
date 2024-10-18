"""
The Llama model configurations and weight downloading utilities.
adopted from opt_config.py
"""

import dataclasses
import glob
import os
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "Llama-2-7b-hf"
    hf_token: str = ''
    hidden_act: str = "silu"
    input_dim: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    n_head: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-05
    dtype: type = np.float16
    pad_token_id: int = 2
    vocab_size: int = 32000

    def model_bytes(self):
        h = self.input_dim
        intermediate = self.intermediate_size
        n_head = self.n_head
        head_dim = h // n_head
        return 2 * (self.vocab_size * h +
        self.num_hidden_layers * (
        # self-attention
        3 * h * h + h * h + head_dim // 2 +
        # mlp
        3 * h * intermediate +
        # layer norm
        2 * h) +
        # head
        h + self.vocab_size * h)

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]

    if "-chat" in name:
        arch_name = name.replace("-chat", "")
    else:
        arch_name = name

    if arch_name == "Llama-2-7b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=4096, intermediate_size=11008, n_head=32,
                             num_hidden_layers=32, num_key_value_heads=32
                             )
    elif arch_name == "Llama-2-13b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=5120, intermediate_size=13824, n_head=40,
                             num_hidden_layers=40, num_key_value_heads=40
                             )
    elif arch_name == "Llama-2-70b-hf":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=8192, intermediate_size=28672, n_head=64,
                             num_hidden_layers=80, num_key_value_heads=8
                             )
    elif arch_name == "Meta-Llama-3.1-8B-Instruct":
        config = LlamaConfig(name=name, hf_token=kwargs.get('hf_token'),
                             input_dim=4096, intermediate_size=14336, n_head=32,
                             num_hidden_layers=32, num_key_value_heads=8, 
                             vocab_size=128256, max_position_embeddings=131072)
    else:
        raise ValueError(f"Invalid model name: {name}")
    print("Padding token id for model: ", config.pad_token_id)
    return dataclasses.replace(config, **kwargs)

def check_format(directory):
    has_bin_file = False
    has_np_folder = False
    has_tensor_file = False
    
    for root, dirs, files in os.walk(directory):
        if any(file.endswith('.bin') for file in files):
            has_bin_file = True
        if any(folder.endswith('np') for folder in dirs):
            has_np_folder = True
        if any(file.endswith('.safetensors') for file in files):
            has_tensor_file = True
    
    has_model_file = has_bin_file or has_tensor_file

    if has_model_file and not has_np_folder:
        # print(f"There are model files and no 'np' ending folders in {directory}. Need format conversion.")
        return (True, False)
    elif not has_model_file and not has_np_folder:
        # print(f"No model files and np 'np' ending folders found in {directory}. Start downloading")
        return (False, False)
    elif has_np_folder:
        # print(f"Found folders ending with 'np' in {directory}.")
        return (None, True)

def convert_llama_weights(model_name, path):
    import torch
    print(f"HuggingFace format llama model exist. "
          f"But FlexGen require to convert to numpy format. "
          f"Start format conversion.")
    folder = path
    bin_files = glob.glob(os.path.join(folder, "*.bin"))
    tensor_files = glob.glob(os.path.join(folder, "*.safetensors"))
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    if len(tensor_files) != 0:
        from safetensors.torch import load_file
        for tensor_file in tqdm(tensor_files, desc="Conver format"):
            state = load_file(tensor_file)
            for name, param in tqdm(state.items(), leave=False):
                name = name.replace("model.", "")
                param_path = os.path.join(path, name)
                if param.dtype == torch.bfloat16:
                    param = param.float()
                with open(param_path, "wb") as f:
                    np.save(f, param.cpu().detach().numpy())
            
    elif len(bin_files) != 0:
        for bin_file in tqdm(bin_files, desc="Convert format"):
            state = torch.load(bin_file, map_location='cuda:0')
            for name, param in tqdm(state.items(), leave=False):
                name = name.replace("model.", "")
                param_path = os.path.join(path, name)
                with open(param_path, "wb") as f:
                    np.save(f, param.cpu().detach().numpy())
    

def download_llama_weights(model_name, path, hf_token):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    hf_model_name = "meta-llama/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin", token=hf_token)
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1]
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file, map_location='cuda:0')
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())