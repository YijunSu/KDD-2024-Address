import os
import sys
import torch
import torch.nn as nn
from loguru import logger
from transformers import BertConfig, BertModel
from const import root_workspace, pretrained_dir


class GeographicalAttentionNetwork(nn.Module):
    def __init__(
            self,
            pretrained_model_name="chinese_wwm_ext_pytorch",
            with_entity_type=False,
            entity_pooler_insert_layer=None,
            output_pooling="last-avg"):
        super(GeographicalAttentionNetwork, self).__init__()
        if pretrained_model_name:
            # 加载中文预训练模型
            self.config = BertConfig.from_pretrained(
                os.path.join(pretrained_dir, pretrained_model_name, "bert_config.json"))
            self.pretrained_bert = BertModel.from_pretrained(
                os.path.join(pretrained_dir, pretrained_model_name), config=self.config)
        else:
            # 初始化bert模型
            self.config = BertConfig(
                vocab_size=21128,
                num_hidden_layers=4,
                num_attention_heads=4,
                hidden_size=256,
                intermediate_size=256 * 4,
                type_vocab_size=16,
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.2,
            )
            self.pretrained_bert = BertModel(self.config)

        # 加入地理实体类别embedding, 共16种地理实体类别
        self.with_entity_type = with_entity_type
        if self.with_entity_type:
            self.config.type_vocab_size = 16
            self.pretrained_bert.embeddings.token_type_embeddings = nn.Embedding(16, self.config.hidden_size)

        # 加入地理实体池化层
        self.entity_pooler_insert_layer = entity_pooler_insert_layer
        if self.entity_pooler_insert_layer and self.entity_pooler_insert_layer > 0:
            self.layer_after_entity_pooler = self.pretrained_bert.encoder.layer[entity_pooler_insert_layer:]
            del self.pretrained_bert.encoder.layer[entity_pooler_insert_layer:]
        self.output_pooling = output_pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 不注入地理实体信息
        if not self.with_entity_type:
            bert_out = self.pretrained_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                output_hidden_states=True)
        else:
            bert_out = self.pretrained_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True)
        first = bert_out.hidden_states[1].transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]
        last = bert_out.last_hidden_state.transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]

        # 添加地理实体池化层
        if self.entity_pooler_insert_layer:
            token_type_ids = token_type_ids.unsqueeze(dim=2)  # [batch, seq_len, 1] = [bs, 64, 1]
            etty_pooled_out = []
            # Entity Pooling：对0~15类实体进行平均池化操作, 16为[CLS]、[SEP]以及0等填充token
            for etty_type in range(self.config.type_vocab_size - 1):
                mask = torch.where(token_type_ids == etty_type, 1., 0.)  # [batch, seq_len, 1] = [bs, 64, 1]
                len_ = torch.sum(mask, dim=1, dtype=torch.float32) + 0.000001  # [batch, 1]
                # last_hidden_state: [batch, seq_len, hidden_size]  -> state_avg: [batch, hidden_size] = [bs, 768]
                state_avg = torch.sum(bert_out.last_hidden_state * mask, dim=1) / len_
                etty_pooled_out.append(state_avg)
            etty_pooled_out = torch.stack(etty_pooled_out,
                                          dim=0)  # [type_etty_size, batch, hidden_size] = [15, bs, 768]
            etty_pooled_out = etty_pooled_out.transpose(0, 1)  # [batch, type_etty_size, hidden_size] = [bs, 15, 768]
            for lay_module in self.layer_after_entity_pooler:
                etty_pooled_out = lay_module(etty_pooled_out)
                etty_pooled_out = etty_pooled_out[0]
            last = etty_pooled_out.transpose(1, 2)  # [batch, hidden_size, type_etty_size] = [bs, 768, 15]

        # 池化方法
        if self.output_pooling == "last-avg":
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.output_pooling == "first-last-avg":
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


class SimCSE(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, use_token_type, pooling='last-avg'):
        super(SimCSE, self).__init__()
        self.config = BertConfig(
            vocab_size=21128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            type_vocab_size=16)
        self.bert = BertModel(self.config)
        # self.config = BertConfig.from_pretrained(f"{root_workspace}/pretrained_model/bert_wwm_ext_pytorch")
        # self.bert = BertModel.from_pretrained(f"{root_workspace}/pretrained_model/bert_wwm_ext_pytorch")
        self.use_token_type = use_token_type
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        if not self.use_token_type:
            token_type_ids = None

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


class EntitySimCSE(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_entity_layers, pooling='last-avg'):
        super(EntitySimCSE, self).__init__()
        self.config = BertConfig(
            vocab_size=21128,
            num_hidden_layers=num_hidden_layers - num_entity_layers,
            num_attention_heads=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            type_vocab_size=16)
        self.word_bert = BertModel(self.config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.intermediate_size,
            activation=self.config.hidden_act)
        self.etty_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_entity_layers)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        word_out = self.word_bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        token_type_ids = token_type_ids.unsqueeze(dim=2)  # [batch, seq_len, 1] = [bs, 64, 1]
        etty_pooled_out = []
        # Entity Pooling：对0~15类实体进行平均池化操作, 16为[CLS]、[SEP]以及0等填充token
        for etty_type in range(self.config.type_vocab_size - 1):
            mask = torch.where(token_type_ids == etty_type, 1., 0.)  # [batch, seq_len, 1] = [bs, 64, 1]
            len_ = torch.sum(mask, dim=1, dtype=torch.float32) + 0.000001  # [batch, 1]
            # word_out.last_hidden_state: [batch, seq_len, hidden_size]  -> state_avg: [batch, hidden_size] = [bs, 768]
            state_avg = torch.sum(word_out.last_hidden_state * mask, dim=1) / len_
            etty_pooled_out.append(state_avg)
        etty_pooled_out = torch.stack(etty_pooled_out, dim=0)  # [type_etty_size, batch, hidden_size] = [15, bs, 768]
        # NOTE: transformer只能输入[seq_len, batch, hidden_size]形式的数据
        etty_out = self.etty_transformer(etty_pooled_out)  # [type_etty_size, batch, hidden_size] = [15, bs, 768]
        etty_out = etty_out.transpose(0, 1)  # [batch, type_etty_size, hidden_size] = [bs, 15, 768]

        if self.pooling == 'last-avg':
            last = etty_out.transpose(1, 2)  # [batch, hidden_size, type_etty_size] = [bs, 768, 15]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = word_out.hidden_states[1].transpose(1, 2)  # [batch, hidden_size, seq_len] = [bs, 768, 64]
            last = etty_out.transpose(1, 2)  # [batch, hidden_size, type_etty_size] = [bs, 768, 15]
            first_avg = torch.avg_pool1d(first, kernel_size=first.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


# def init_models(n_parties, model_type, num_hidden_layers, hidden_size, num_entity_layers, use_token_type):
#     if model_type == "SimCSE":
#         models = [
#             SimCSE(
#                 num_hidden_layers=num_hidden_layers,
#                 hidden_size=hidden_size,
#                 use_token_type=use_token_type
#             ) for _ in range(n_parties)
#         ]
#     elif model_type == "EntitySimCSE":
#         models = [
#             EntitySimCSE(
#                 num_hidden_layers=num_hidden_layers,
#                 hidden_size=hidden_size,
#                 num_entity_layers=num_entity_layers
#             ) for _ in range(n_parties)
#         ]
#     else:
#         raise ValueError(f"Error model type {model_type}")
#     return models


def init_models(n_parties, pretrained_model_name, with_entity_type, entity_pooler_insert_layer):
    models = [
        GeographicalAttentionNetwork(
            pretrained_model_name=pretrained_model_name,
            with_entity_type=with_entity_type,
            entity_pooler_insert_layer=entity_pooler_insert_layer
        ) for _ in range(n_parties)
    ]
    logger.info(f"init models: \n{models}")
    return models