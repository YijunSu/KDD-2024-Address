import os
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_workspace = os.path.split(os.path.realpath(__file__))[0]
config_dir = os.path.join(root_workspace, "configs")
data_dir = os.path.join(root_workspace, "datasets")
log_dir = os.path.join(root_workspace, "logs")
board_dir = os.path.join(root_workspace, "boards")
pretrained_dir = os.path.join(root_workspace, "pretrained_model")
model_dir = os.path.join(root_workspace, "saved_models")
