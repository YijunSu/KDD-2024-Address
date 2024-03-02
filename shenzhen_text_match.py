import torch
import torchmetrics
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torchmetrics.functional.regression import cosine_similarity

from const import *
from data import _etty_padding


tokenizer = BertTokenizer(f'{root_workspace}/pretrained_model/bert_wwm_ext_pytorch/vocab.txt')


class ShenzhenAddress(Dataset):
    def __init__(self, data, tokenizer=tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # 每一行为 [address1	address1_ger	address2	address2_ger	label]
        line = self.data[index].split("\t")
        address1, address1_etty = line[0], [line[1].split(",")]
        address2, address2_etty = line[2], [line[3].split(",")]
        label = int(line[4])
        
        address1_inputs = self.tokenizer(address1, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        address1_inputs["token_type_ids"] = _etty_padding(address1_etty, max_length=self.max_length, pad_value=15)
        address2_inputs = self.tokenizer(address2, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        address2_inputs["token_type_ids"] = _etty_padding(address2_etty, max_length=self.max_length, pad_value=15)
        
        return address1_inputs, address2_inputs, torch.tensor(label)
    
    def __len__(self):
        return len(self.data)


def load_data(path):
    with open(path, "r", encoding="utf8") as f:
        f_bar = tqdm(f)
        f_bar.set_description(f"Loading data from {path}")
        return [line.strip() for i, line in enumerate(f_bar) if i != 0]


if __name__ == '__main__':
    # load test data
    test_file_name = "shenzhen_address_match_data/shenzen_ner_test.txt"
    test_file_path = os.path.join(data_dir, test_file_name)
    test_ds = ShenzhenAddress(load_data(test_file_path))
    test_dl = DataLoader(test_ds, batch_size=64)
    model = torch.load("saved_models/2022-05-30/Federated/Transformer+GE-type+Entity-Pooler/Sup-SimCSE/17:39:48-Federated_Transformer+GE-type+Entity-Pooler_Sup-SimCSE-global.pt")
    model.to(device)
    model.eval()

    for threshold in np.arange(0.6, 1.01, 0.02):
        Accuracy = torchmetrics.Accuracy(threshold=threshold).to(device)
        Precision = torchmetrics.Precision(threshold=threshold, ignore_index=0).to(device)
        Recall = torchmetrics.Recall(threshold=threshold, ignore_index=0).to(device)
        F1Score = torchmetrics.F1Score(threshold=threshold, ignore_index=0).to(device)
        AUROC = torchmetrics.AUROC().to(device)
        SpearmanCorrCoef = torchmetrics.SpearmanCorrCoef().to(device)
        with torch.no_grad():
            for batch_idx, source in enumerate(test_dl):
                address1_inputs, address2_inputs, label = source
                batch_num = address1_inputs.get("input_ids").shape[0]
                
                address1_input_ids = address1_inputs.get("input_ids").view(batch_num, -1).to(device)
                address1_attention_mask = address1_inputs.get("attention_mask").view(batch_num, -1).to(device)
                address1_token_type_ids = address1_inputs.get("token_type_ids").view(batch_num, -1).to(device)
#                 logger.info(f"address1: input_ids.shape={address1_input_ids.shape}, attention_mask.shape={address1_attention_mask.shape}, token_type_ids.shape={address1_token_type_ids.shape}")
                
                address2_input_ids = address2_inputs.get("input_ids").view(batch_num, -1).to(device)
                address2_attention_mask = address2_inputs.get("attention_mask").view(batch_num, -1).to(device)
                address2_token_type_ids = address2_inputs.get("token_type_ids").view(batch_num, -1).to(device)
#                 logger.info(f"address2: input_ids.shape={address2_input_ids.shape}, attention_mask.shape={address2_attention_mask.shape}, token_type_ids.shape={address2_token_type_ids.shape}")
                
                address1_outputs = model(address1_input_ids, address1_attention_mask, address1_token_type_ids)
                address2_outputs = model(address2_input_ids, address2_attention_mask, address2_token_type_ids)
#                 logger.info(f"address1_outputs.shape={address1_outputs.shape}")
#                 logger.info(f"address2_outputs.shape={address2_outputs.shape}")
                
                sim = cosine_similarity(address1_outputs, address2_outputs, 'none').view(batch_num, -1)
                y_prob = sim
                y_pred = torch.where(y_prob > threshold, 1, 0).to(device)
                y_true = label.view(batch_num, -1).to(device)
#                 logger.info(f"y_true.shape={y_true.shape}, y_prob.shape={y_prob.shape}, y_pred.shape={y_pred.shape}")
                
                # 指标计算
                Accuracy(y_pred, y_true)
                Precision(y_pred, y_true)
                Recall(y_pred, y_true)
                F1Score(y_pred, y_true)
                AUROC(y_prob, y_true)
                SpearmanCorrCoef(y_prob, y_true.type(torch.float32))

            acc, pre, recall, f1, roc, spear = Accuracy.compute(), Precision.compute(), Recall.compute(), F1Score.compute(), AUROC.compute(), SpearmanCorrCoef.compute()
            Accuracy.reset(), Precision.reset(), Recall.reset(), F1Score.reset(), AUROC.reset(), SpearmanCorrCoef.reset()
            logger.info("------------------------------Evaluation------------------------------")
            logger.info(f"All average metrics on evaluation dataset, threshold is {threshold}.")
            logger.info(f"accuracy={acc}, precision={pre}, recall={recall}, f1={f1}, auroc={roc}, spearman={spear}")
            logger.info("------------------------------Evaluation------------------------------")
