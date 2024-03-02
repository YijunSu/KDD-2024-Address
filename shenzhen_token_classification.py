from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from const import *


tokenizer = BertTokenizer(f'{root_workspace}/pretrained_model/bert_wwm_ext_pytorch/vocab.txt')


class ShenzhenAddress(Dataset):
    def __init__(self, data, tokenizer=tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # 每一行为 [address1	address1_seg	address1_tag	address2	address2_seg	address2_tag	label]
        line = self.data[index].split("\t")
        logger.info(line)
        address1, address2, label = line[0], line[3], int(line[6])
        logger.info(f"{address1}: {type(address1)}")
        address1_inputs = self.tokenizer(address1, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        address2_inputs = self.tokenizer(address2, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return address1, address1_inputs, address2, address2_inputs, torch.tensor(label)
    
    def __len__(self):
        return len(self.data)


def load_data(path):
    with open(path, "r", encoding="utf8") as f:
        f_bar = tqdm(f)
        f_bar.set_description(f"Loading data from {path}")
        return [line.strip() for i, line in enumerate(f_bar) if i != 0]


if __name__ == '__main__':
    # load test data
    test_file_name = "shenzhen_address_match_data/shenzhen_ner_cls_del_test_16000.txt"
    test_file_path = os.path.join(data_dir, test_file_name)
    test_ds = ShenzhenAddress(load_data(test_file_path))
    test_dl = DataLoader(test_ds, batch_size=1)
    model = torch.load("saved_models/geographical_entity_recognition_residence.pt")
    model.to(device)
    model.eval()
    
    ner_test_file_name = "shenzhen_address_match_data/shenzen_ner_test.txt"
    ner_test_file_path = os.path.join(data_dir, ner_test_file_name)
    with open(test_file_path, "r", encoding="utf8") as reader, open(ner_test_file_path, "w", encoding="utf8") as writer:
        f_bar = tqdm(reader)
        f_bar.set_description(f"Loading data from {test_file_path}")
        for i, line in enumerate(f_bar):
            if i == 0:
                continue
            address1, _, _, address2, _, _, label = line.strip().split("\t")
            address1_inputs = tokenizer(address1, max_length=64, truncation=True, return_tensors='pt').to(device)
            address2_inputs = tokenizer(address2, max_length=64, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                address1_token_types = model(**address1_inputs).logits.argmax(-1)
                address2_token_types = model(**address2_inputs).logits.argmax(-1)
            address1_token_types = address1_token_types[0].tolist()[1:-1]
            address2_token_types = address2_token_types[0].tolist()[1:-1]
            logger.info(f"address1_tokens:{tokenizer.tokenize(address1)}, address1_token_types: {address1_token_types}")
            logger.info(f"address2_tokens:{tokenizer.tokenize(address2)}, address2_token_types: {address2_token_types}")
            temp = [address1, ",".join(map(str, address1_token_types)), address2, ",".join(map(str, address2_token_types)), str(label)]
            logger.info("\t".join(temp))
            writer.write("\t".join(temp)+"\n")
