import torch
import torchmetrics
import numpy as np
from loguru import logger
from sklearn import metrics
import torch.nn.functional as F
from scipy.stats import spearmanr
from const import device
from torchmetrics.functional.regression import cosine_similarity


def simcse_unsup_loss(h_pred, temperature=0.05):
    """
    无监督损失函数
    h_pred: 模型输出表征向量
    temperature: 温度系数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idxs = torch.arange(h_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    sim = F.cosine_similarity(h_pred.unsqueeze(1), h_pred.unsqueeze(0), dim=2)
    sim = sim - torch.eye(h_pred.shape[0], device=device) * 1e12
    sim = sim / temperature
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def simcse_sup_loss(h_pred, temperature=0.05):
    """
    有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    temperature: 温度系数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    idxs = torch.arange(h_pred.shape[0], device=device)
    used_idxs = torch.where((idxs + 1) % 3 != 0)[0]
    y_true = used_idxs + 1 - used_idxs % 3 * 2
    sim = F.cosine_similarity(h_pred.unsqueeze(1), h_pred.unsqueeze(0), dim=2)
    sim = sim - torch.eye(h_pred.shape[0], device=device) * 1e12
    sim = torch.index_select(sim, 0, used_idxs)
    sim = sim / temperature
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def sup_triplet_loss(h_pred, margin=1):
    """
    Sentence-BERT：有监督三元组损失
    :param h_pred: bert的输出, [batch_size * 3, 768]
    :param margin:
    :return:
    """
    idx = torch.arange(h_pred.shape[0], device=device).unsqueeze(1)
    anchor = F.normalize(h_pred.index_select(0, torch.where(idx % 3 == 0)[0]), p=2, dim=1)
    positive = F.normalize(h_pred.index_select(0, torch.where(idx % 3 == 1)[0]), p=2, dim=1)
    negative = F.normalize(h_pred.index_select(0, torch.where(idx % 3 == 2)[0]), p=2, dim=1)
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss.mean()


def local_trainer(comm_round, party_id, model, train_dl, dev_dl, epochs, lr, train_type, model_type, dev_interval, writer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if train_type == "supervised":
        if model_type == "SimCSE":
            criterion = simcse_sup_loss
        elif model_type == "Triplet":
            criterion = sup_triplet_loss
    elif train_type == 'unsupervised':
        if model_type == "SimCSE":
            criterion = simcse_unsup_loss

    # 未训练时先对模型验证
    if comm_round == 0:
        local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear = evaluate(
            model, dev_dl, criterion)
        local_train_loss = local_dev_loss
        return local_train_loss, local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear

    model.train()
    logger.info(f"Party {party_id} start local training")
    local_train_loss_collector = []
    for epoch in range(epochs):
        batch_loss_collector = []

        for batch_idx, source in enumerate(train_dl, start=1):
            real_batch_num = source.get('input_ids').shape[0]
            # sup: batch_size * 3, unsup: batch_size * 6
            if train_type == 'supervised':
                input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(device)
                attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(device)
                token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(device)
            else:
                input_ids = source.get('input_ids').view(real_batch_num * 6, -1).to(device)
                attention_mask = source.get('attention_mask').view(real_batch_num * 6, -1).to(device)
                token_type_ids = source.get('token_type_ids').view(real_batch_num * 6, -1).to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss_collector.append(loss.item())

            cur_batch = ((comm_round - 1) * epochs + epoch) * len(train_dl) + batch_idx
            if cur_batch % 10 == 0:
                logger.info(f"Batch {cur_batch}: loss={loss.item()}")

            if cur_batch % 100 == 0:
                batch_loss = sum(batch_loss_collector) / len(batch_loss_collector)
                batch_loss_collector = []
                local_train_loss_collector.append(batch_loss)
                writer.add_scalar(f'{party_id}-Train/Loss', batch_loss, cur_batch)

            # if cur_batch % dev_interval == 0:
            #     dev_loss, dev_spear, dev_acc = evaluate(model, dev_dl, criterion)
            #     model.train()
            #     local_dev_loss_collector.append(dev_loss)
            #     local_dev_spear_collector.append(dev_spear)
            #     local_dev_acc_collector.append(dev_acc)
        if batch_loss_collector:
            local_train_loss_collector.append(sum(batch_loss_collector) / len(batch_loss_collector))

    logger.info(f"Party {party_id} finish local training")
    local_train_loss = sum(local_train_loss_collector) / len(local_train_loss_collector)

    # local dev metrics
    local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear = evaluate(model, dev_dl, criterion)

    return local_train_loss, local_dev_loss, local_dev_acc, local_dev_pre, local_dev_recall, local_dev_f1, local_dev_roc, local_dev_spear


# def eval(model, dev_dl, criterion, threshold=0.5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#     loss_all, pred_all, prob_all, label_all = [], [], [], []
#     with torch.no_grad():
#         for batch_idx, source in enumerate(dev_dl):
#             real_batch_num = source.get('input_ids').shape[0]
#             input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(device)
#             attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(device)
#             token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(device)
#             out = model(input_ids, attention_mask, token_type_ids)
#             idx = torch.arange(out.shape[0], device=device).unsqueeze(1)
#             anchor = out.index_select(0, torch.where(idx % 3 == 0)[0])
#             positive = out.index_select(0, torch.where(idx % 3 == 1)[0])
#             negative = out.index_select(0, torch.where(idx % 3 == 2)[0])
#             sim_1 = F.cosine_similarity(anchor, positive, dim=-1)
#             sim_0 = F.cosine_similarity(anchor, negative, dim=-1)
#             y_prob = torch.cat([sim_1, sim_0], dim=0)
#             y_pred = torch.where(y_prob > threshold, 1, 0)
#             y_true = torch.cat([torch.ones_like(sim_1), torch.zeros_like(sim_0)], dim=0)
#
#             pred_all.append(y_pred.cpu())
#             prob_all.append(y_prob.cpu())
#             label_all.append(y_true.cpu())
#             loss_all.append(simcse_sup_loss(out).item())
#
#         loss = sum(loss_all) / len(loss_all)
#         prob_all = torch.cat(prob_all, dim=0).numpy()
#         pred_all = torch.cat(pred_all, dim=0).numpy()
#         label_all = torch.cat(label_all, dim=0).numpy()
#         acc, pre, recall, f1, roc, spear = classification_metrics(label_all, pred_all, prob_all)
#     model.train()
#     return loss, acc, pre, recall, f1, roc, spear
#
#
# def classification_metrics(y_true, y_pred, y_prob):
#     accuracy = metrics.accuracy_score(y_true, y_pred)
#     precision = metrics.precision_score(y_true, y_pred)
#     recall = metrics.recall_score(y_true, y_pred)
#     f1 = metrics.f1_score(y_true, y_pred)
#     roc_auc = metrics.roc_auc_score(y_true, y_prob)
#     spear = spearmanr(y_true, y_prob).correlation
#     return accuracy, precision, recall, f1, roc_auc, spear


def evaluate(model, dev_dl, criterion, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    Accuracy = torchmetrics.Accuracy(threshold=threshold).to(device)
    Precision = torchmetrics.Precision(threshold=threshold, ignore_index=0).to(device)
    Recall = torchmetrics.Recall(threshold=threshold, ignore_index=0).to(device)
    F1Score = torchmetrics.F1Score(threshold=threshold, ignore_index=0).to(device)
    AUROC = torchmetrics.AUROC().to(device)
    SpearmanCorrCoef = torchmetrics.SpearmanCorrCoef().to(device)
    loss, batch_loss_count = 0, 0
    with torch.no_grad():
        for batch_idx, source in enumerate(dev_dl):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(device)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(device)
            out = model(input_ids, attention_mask, token_type_ids)
            idx = torch.arange(out.shape[0], device=device).unsqueeze(1)
            # 三元组 (anchor, positive, negetive)
            anchor = out.index_select(0, torch.where(idx % 3 == 0)[0])
            positive = out.index_select(0, torch.where(idx % 3 == 1)[0])
            negative = out.index_select(0, torch.where(idx % 3 == 2)[0])

            # y_true、y_pred、y_prob
            sim_1 = cosine_similarity(anchor, positive, 'none')
            sim_0 = cosine_similarity(anchor, negative, 'none')
            y_prob = torch.cat([sim_1, sim_0], dim=0).to(device)
            y_pred = torch.where(y_prob > threshold, 1, 0).to(device)
            y_true = torch.cat([torch.ones_like(sim_1, dtype=torch.int), torch.zeros_like(sim_0, dtype=torch.int)],
                               dim=0).to(device)

            # 指标计算
            batch_loss = criterion(out).item()
            loss += batch_loss
            batch_loss_count += 1
            Accuracy(y_pred, y_true)
            Precision(y_pred, y_true)
            Recall(y_pred, y_true)
            F1Score(y_pred, y_true)
            AUROC(y_prob, y_true)
            SpearmanCorrCoef(y_prob, y_true.type(torch.float32))

        loss /= batch_loss_count
        acc = round(Accuracy.compute().item(), 4)
        pre, recall, f1 = round(Precision.compute().item(), 4), round(Recall.compute().item(), 4), round(F1Score.compute().item(), 4)
        roc, spear = round(AUROC.compute().item(), 4), round(SpearmanCorrCoef.compute().item(), 4)
        Accuracy.reset(), Precision.reset(), Recall.reset(), F1Score.reset(), AUROC.reset(), SpearmanCorrCoef.reset()
        logger.info("------------------------------Evaluation------------------------------")
        logger.info(f"All average metrics on evaluation dataset, criterion is {criterion}, threshold is {threshold}.")
        logger.info(f"loss={loss}, accuracy={acc}, precision={pre}, recall={recall}, f1={f1}, auroc={roc}, spearman={spear}")
        logger.info("------------------------------Evaluation------------------------------")
    model.train()
    return loss, acc, pre, recall, f1, roc, spear