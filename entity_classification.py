import torch
import torchmetrics
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from const import *
from data import get_dataloader
from utils import mkdir, mknod


def evaluate(model, test_dl):
    model.eval()
    Accuracy = torchmetrics.Accuracy().to(device)
    Precision = torchmetrics.Precision().to(device)
    Recall = torchmetrics.Recall().to(device)
    F1Score = torchmetrics.F1Score().to(device)
    with torch.no_grad():
        dev_loss = 0
        for batch_idx, source in enumerate(test_dl):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(device)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(device)
            outputs = model(input_ids, attention_mask, labels=token_type_ids)
            y_true = token_type_ids.flatten()
            y_pred = outputs.logits.argmax(-1).flatten()
            Accuracy(y_pred, y_true)
            Precision(y_pred, y_true)
            Recall(y_pred, y_true)
            F1Score(y_pred, y_true)
            predicted_token_class_ids = outputs.logits.argmax(-1)
            dev_loss += outputs.loss
    loss = dev_loss / len(test_dl)
    acc, precision, recall, f1 = Accuracy.compute(), Precision.compute(), Recall.compute(), F1Score.compute()
    return loss, acc, precision, recall, f1


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(pretrained_dir, 'chinese_wwm_ext_pytorch', 'vocab.txt'))
    config = BertConfig.from_pretrained(
        os.path.join(pretrained_dir, 'chinese_wwm_ext_pytorch', "bert_config.json"))
    config.num_labels = 16
    model = BertForTokenClassification.from_pretrained(
        os.path.join(pretrained_dir, 'chinese_wwm_ext_pytorch'), config=config)
    model_path = "saved_models/geographical_entity_recognition_residence.pt"
    mknod(model_path)
    train_dl = get_dataloader("datasets/triplet_residence_train_50k.txt", "supervised")
    test_dl = get_dataloader("datasets/triplet_residence_dev_5k.txt", "supervised")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 20
    for epoch in range(epochs):
        model.to(device)
        model.train()
        train_loss = 0
        for batch_idx, source in enumerate(train_dl, start=1):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(device)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(device)
            outputs = model(input_ids, attention_mask, labels=token_type_ids)
            loss = outputs.loss
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {batch_idx}: {loss}")
        dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(model, test_dl)
        print(f"-----------------------Evaluation-----------------------")
        print(f"Epoch {epoch}: loss={dev_loss}, acc={dev_acc}, precision={dev_precision}, recall={dev_recall}, f1={dev_f1}")
        print(f"-----------------------Evaluation-----------------------")
        torch.save(model, model_path)



