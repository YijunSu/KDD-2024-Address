import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

root_workspace = os.path.split(os.path.realpath(__file__))[0]
tokenizer = BertTokenizer(f'{root_workspace}/pretrained_model/bert_wwm_ext_pytorch/vocab.txt')


def _etty_padding(entities, max_length=64, pad_value=15):
    padded_entities = []
    for ele in entities:
        ele = [int(type_) for type_ in ele]
        padded_entities.append(F.pad(input=torch.tensor(ele), pad=(1, max_length - len(ele) - 1), value=pad_value))
    return torch.stack(padded_entities, dim=0)


class UnsupervisedTrain(Dataset):
    def __init__(self, data, tokenizer=tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # 每一行为地址三元组 [origin_text, origin_etty, positive_text, positive_etty, negative_text, negative_etty]
        line = self.data[index].split("\t")
        # 无监督只取第一列anchor
        anchor_text, anchor_etty = line[0].split(" "), line[1].split(" ")
        positive_text, positive_etty = line[2].split(" "), line[3].split(" ")
        negative_text, negative_etty = line[4].split(" "), line[5].split(" ")
        anchor_dict = self.tokenizer(
            [anchor_text, anchor_text, positive_text, positive_text, negative_text, negative_text],
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        anchor_dict["token_type_ids"] = _etty_padding(
            entities=[anchor_etty, anchor_etty, positive_etty, positive_etty, negative_etty, negative_etty],
            max_length=self.max_length,
            pad_value=15)
        return anchor_dict

    def __len__(self):
        return len(self.data)


class SupervisedTrain(Dataset):
    def __init__(self, data, tokenizer=tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # 每一行为地址三元组 [origin_text, origin_etty, positive_text, positive_etty, negative_text, negative_etty]
        line = self.data[index].split("\t")
        triplet_text = []
        triplet_etty = []
        for i in range(len(line)):
            if i % 2 == 0:
                triplet_text.append(line[i].split(" "))
            else:
                triplet_etty.append(line[i].split(" "))

        # tokenize address text & padding address etty
        triplet_dict = self.tokenizer(triplet_text, is_split_into_words=True, max_length=self.max_length,
                                      truncation=True,
                                      padding='max_length', return_tensors='pt')
        triplet_etty = _etty_padding(triplet_etty, max_length=self.max_length, pad_value=15)
        triplet_dict["token_type_ids"] = triplet_etty
        return triplet_dict

    def __len__(self):
        return len(self.data)


def load_data(path):
    with open(path, "r", encoding="utf8") as f:
        f_bar = tqdm(f)
        return [line.strip() for _, line in enumerate(f_bar)]


def get_dataset(path, data_type):
    data = load_data(path)
    if data_type == "supervised":
        dataset = SupervisedTrain(data)
    elif data_type == "unsupervised":
        dataset = UnsupervisedTrain(data)
    elif data_type == "eval":
        dataset = SupervisedTrain(data)
    return dataset


def get_dataloader(path, data_type, batch_size=64):
    ds = get_dataset(path, data_type)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl


def init_data_loaders(data_dir, train_files, dev_files, train_type, batch_size, split_size=0.8):
    train_dls, train_nums, dev_dls, dev_nums = [], [], [], []
    for i in range(len(train_files)):
        if dev_files:
            train_path = os.path.join(data_dir, train_files[i])
            train_ds = get_dataset(path=train_path, data_type=train_type)
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            train_num = len(train_ds)
            dev_path = os.path.join(data_dir, dev_files[i])
            dev_ds = get_dataset(path=dev_path, data_type="eval")
            dev_dl = DataLoader(dev_ds, batch_size=batch_size)
            dev_num = len(dev_ds)
        else:
            full_path = os.path.join(data_dir, train_files[i])
            full_ds = get_dataset(path=full_path, data_type=train_type)
            train_num = int(len(full_ds) * split_size)
            dev_num = len(full_ds) - train_num
            train_ds, dev_ds = torch.utils.data.random_split(full_ds, [train_num, dev_num])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            dev_dl = DataLoader(dev_ds, batch_size=batch_size)
        train_dls.append(train_dl)
        train_nums.append(train_num)
        dev_dls.append(dev_dl)
        dev_nums.append(dev_num)
    return train_dls, train_nums, dev_dls, dev_nums
