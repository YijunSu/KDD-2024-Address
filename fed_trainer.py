import os
import sys
import yaml
import time
import torch
import random
import argparse
import datetime
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from utils import mkdir, mknod
from data import init_data_loaders
from model import init_models
from local_trainer import local_trainer

# mkdir [configs, datasets, logs, boards, pretrained_model, saved_models]
root_workspace = os.path.split(os.path.realpath(__file__))[0]
config_dir = os.path.join(root_workspace, "configs")
data_dir = os.path.join(root_workspace, "datasets")
log_dir = os.path.join(root_workspace, "logs")
board_dir = os.path.join(root_workspace, "boards")
pretrained_dir = os.path.join(root_workspace, "pretrained_model")
model_dir = os.path.join(root_workspace, "saved_models")
mkdir(config_dir), mkdir(data_dir), mkdir(log_dir), mkdir(pretrained_dir), mkdir(model_dir)
needed_dirs = [config_dir, data_dir, log_dir, pretrained_dir, model_dir]
sys.path.extend(needed_dirs)
logger.debug(f"root_workspace={root_workspace}")
logger.debug(f"directories: {needed_dirs}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="default.yaml", help='config.yaml file name')
    parser.add_argument("--data_dir", default=None)
    parser.add_argument('--model_dir', default=None)
    args = parser.parse_args()
    config_file = args.config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "r", encoding="utf8") as f:
        params = yaml.safe_load(f)
    logger.debug(f"model params:\n {params}")

    # parse params
    n_parties = params["n_parties"]
    local_train_files = params["local_train_files"]
    local_dev_files = params["local_dev_files"]
    n_comm_round = params["n_comm_round"]
    local_epochs = params["local_epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    dev_interval = params["dev_interval"]
    early_stop_rounds = params["early_stop_rounds"]
    init_seed = params["init_seed"]
    train_type = params["train_type"]
    model_type = params["model_type"]
    num_hidden_layers = params["num_hidden_layers"]
    hidden_size = params["hidden_size"]
    num_entity_layers = params["num_entity_layers"]
    use_token_type = params["use_token_type"]

    # init random seed
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init log_path, board_path, local_model_paths
    cur_date = datetime.date.today().strftime('%Y-%m-%d')
    temp_dir = os.path.join(cur_date, os.path.split(config_file)[0])
    config_name = os.path.split(config_file)[-1].split(".")[0]
    log_dir = os.path.join(log_dir, temp_dir)
    board_dir = os.path.join(board_dir, temp_dir)
    model_dir = os.path.join(model_dir, temp_dir)
    mkdir(log_dir), mkdir(board_dir), mkdir(model_dir)
    cur_time = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')

    log_name = f"{config_name}-{cur_time}"
    log_path = os.path.join(log_dir, log_name)
    logger.add(log_path)
    logger.debug(f"log path is {log_path}")

    board_path = os.path.join(board_dir, log_name)
    writer = SummaryWriter(board_path)
    logger.debug(f"tensorboard path is {board_path}")

    local_model_paths = []
    for train_file in local_train_files:
        train_file_name = train_file.split(".")[0]
        model_name = f"{train_file_name}-{config_name}-{cur_time}.pt"
        model_path = os.path.join(model_dir, model_name)
        mknod(model_path)
        torch.save("0", model_path)
        local_model_paths.append(model_path)
    logger.debug(f"local model paths is {local_model_paths}")

    # init data loaders
    local_train_dls, local_train_nums, local_dev_dls, local_dev_nums = \
        init_data_loaders(data_dir, local_train_files, local_dev_files, train_type, batch_size)

    # init models
    local_models = init_models(n_parties, model_type, num_hidden_layers, hidden_size, num_entity_layers, use_token_type)

    # n_round_unimproved = 0
    # best_spear, best_acc = -1, 0
    if n_parties == 1:
        time_start = time.time()
        for round_i in range(n_comm_round):
            # 每5轮保存一下模型
            if round_i % 5 == 0:
                logger.info(f">> round {round_i}: save model to {local_model_paths[0]}")
                for model, path in zip(local_models, local_model_paths):
                    torch.save(model, path)

            model = local_models[0]
            train_dl = local_train_dls[0]
            dev_dl = local_dev_dls[0]
            local_train_loss, local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear = \
                local_trainer(
                    comm_round=round_i,
                    party_id=0,
                    model=model,
                    train_dl=train_dl,
                    dev_dl=dev_dl,
                    epochs=local_epochs,
                    lr=lr,
                    train_type=train_type,
                    dev_interval=dev_interval,
                    writer=writer
                )
            writer.add_scalar(f"Train/Loss", local_train_loss, round_i)
            writer.add_scalar(f"Dev/Loss", local_dev_loss, round_i)
            writer.add_scalar(f"Dev/Spearman", local_dev_spear, round_i)
            writer.add_scalar(f"Dev/Accuracy", local_dev_acc, round_i)
            writer.add_scalar(f"Dev/Precision", local_dev_pre, round_i)
            writer.add_scalar(f"Dev/Recall", local_dev_recall, round_i)
            writer.add_scalar(f"Dev/F1", local_dev_f1, round_i)
            writer.add_scalar(f"Dev/Roc_auc", local_dev_roc, round_i)
            writer.add_scalar(f"Time", (time.time()-time_start) / 60, round_i)
            logger.info(
                f"Round {round_i}:\n"
                f"Train/Loss: {local_train_loss}\n"
                f"Dev/Loss: {local_dev_loss}\n"
                f"Dev/Spearman: {local_dev_spear}\n"
                f"Dev/Accuracy: {local_dev_acc}\n"
                f"Dev/Precision: {local_dev_pre}\n"
                f"Dev/Recall: {local_dev_recall}\n"
                f"Dev/F1: {local_dev_f1}\n"
                f"Dev/Roc_auc: {local_dev_roc}"
            )

            # if local_dev_spear > best_spear or local_dev_acc > best_acc:
            #     best_spear = max(local_dev_spear, best_spear)
            #     best_acc = max(local_dev_acc, best_acc)
            #     n_round_unimproved = 0
            # else:
            #     n_round_unimproved += 1
            #
            # # 早停
            # if n_round_unimproved == early_stop_rounds:
            #     logger.info(f">> early stop at round {round_i}: save model to {local_model_paths[0]}")
            #     for model, path in zip(local_models, local_model_paths):
            #         torch.save(model, path)
            #     break

    else:
        global_model = init_models(1, model_type, num_hidden_layers, hidden_size, num_entity_layers, use_token_type)[0]
        global_model_path = os.path.join(model_dir, f"global-{config_name}-{cur_time}.pt")
        mknod(global_model_path)
        torch.save("0", global_model_path)
        logger.debug(f"global model path is {global_model_path}")
        # fed training
        total_data_nums = sum(local_train_nums)
        fed_avg_weight = [num / total_data_nums for num in local_train_nums]
        time_start = time.time()
        for round_i in range(n_comm_round):
            global_w = global_model.state_dict()
            # 每5轮保存一下模型
            if round_i % 5 == 0:
                logger.info(
                    f">> round {round_i}: save global model to {global_model_path}, local model to {local_model_paths}")
                torch.save(global_model, global_model_path)
                for model, path in zip(local_models, local_model_paths):
                    torch.save(model, path)

            # distribute global model to local
            for model in local_models:
                model.load_state_dict(global_w)

            global_train_loss, global_dev_loss, global_dev_spear,  = [], [], []
            global_dev_acc, global_dev_pre, global_dev_recall, global_dev_f1, global_dev_roc = [], [], [], [], []
            for party_i in range(n_parties):
                model = local_models[party_i]
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                train_dl = local_train_dls[party_i]
                dev_dl = local_dev_dls[party_i]
                local_train_loss, local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear = \
                    local_trainer(
                        comm_round=round_i,
                        party_id=party_i,
                        model=model,
                        train_dl=train_dl,
                        dev_dl=dev_dl,
                        epochs=local_epochs,
                        lr=lr,
                        train_type=train_type,
                        dev_interval=dev_interval,
                        writer=writer)
                global_train_loss.append(local_train_loss)
                global_dev_loss.append(local_dev_loss)
                global_dev_spear.append(local_dev_spear)
                global_dev_acc.append(local_dev_acc)
                global_dev_pre.append(local_dev_pre)
                global_dev_recall.append(local_dev_recall)
                global_dev_f1.append(local_dev_f1)
                global_dev_roc.append(local_dev_roc)
                # fed avg
                local_w = model.state_dict()
                for key in local_w:
                    if party_i == 0:
                        global_w[key] = local_w[key] * fed_avg_weight[party_i]
                    else:
                        global_w[key] += local_w[key] * fed_avg_weight[party_i]
                model.to("cpu")
            global_model.load_state_dict(global_w)
            global_train_loss = sum(global_train_loss) / len(global_train_loss)
            global_dev_loss = sum(global_dev_loss) / len(global_dev_loss)
            global_dev_spear = sum(global_dev_spear) / len(global_dev_spear)
            global_dev_acc = sum(global_dev_acc) / len(global_dev_acc)
            global_dev_pre = sum(global_dev_pre) / len(global_dev_pre)
            global_dev_recall = sum(global_dev_recall) / len(global_dev_recall)
            global_dev_f1 = sum(global_dev_f1) / len(global_dev_f1)
            global_dev_roc = sum(global_dev_roc) / len(global_dev_roc)
            writer.add_scalar("Global-Train/Loss", global_train_loss, round_i)
            writer.add_scalar("Global-Dev/Loss", global_dev_loss, round_i)
            writer.add_scalar("Global-Dev/Spearman", global_dev_spear, round_i)
            writer.add_scalar("Global-Dev/Accuracy", global_dev_acc, round_i)
            writer.add_scalar("Global-Dev/Precision", global_dev_pre, round_i)
            writer.add_scalar("Global-Dev/Recall", global_dev_recall, round_i)
            writer.add_scalar("Global-Dev/F1", global_dev_f1, round_i)
            writer.add_scalar("Global-Dev/Roc_auc", global_dev_roc, round_i)
            writer.add_scalar("Time", (time.time()-time_start) / 60, round_i)
            logger.info(
                f"Round {round_i}:\n"
                f"Global-Train/Losss: {global_train_loss}\n"
                f"Global-Dev/Loss: {global_dev_loss}\n"
                f"Global-Dev/Spearman: {global_dev_spear}\n"
                f"Global-Dev/Accuracy: {global_dev_acc}\n"
                f"Global-Dev/Precision: {global_dev_pre}\n"
                f"Global-Dev/Recall: {global_dev_recall}\n"
                f"Global-Dev/F1: {global_dev_f1}\n"
                f"Global-Dev/Roc_auc: {global_dev_roc}"
            )

            # if global_dev_spear > best_spear or global_dev_acc > best_acc:
            #     best_spear = max(global_dev_spear, best_spear)
            #     best_acc = max(global_dev_acc, best_acc)
            #     n_round_unimproved = 0
            # else:
            #     n_round_unimproved += 1
            #
            # # 早停
            # if n_round_unimproved == early_stop_rounds:
            #     logger.info(f">> early stop at round {round_i}: "
            #                 f"save global model to {global_model_path}, local models to {local_model_paths}")
            #     torch.save(global_model, global_model_path)
            #     for model, path in zip(local_models, local_model_paths):
            #         torch.save(model, path)
            #     break
