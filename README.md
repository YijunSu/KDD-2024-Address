# Federated Contrastive Learning

## Dataset
* datasets/train_triplet_institution_600k.txt
* datasets/train_triplet_residence_600k.txt
* datasets/dev_triplet_institution_600k.txt
* datasets/dev_triplet_residence_600k.txt

## Configs
* supervised/unsupervised：有监督/无监督
* EntitySimCSE/SimCSE: 模型类别
* institution/residence/federated：使用数据

### EntitySimCSE_L_12_E_4_H_768.yaml
* L_12: num_hidden_layers=12
* E_4: entity_hidden_layers=4
* H_768: hidden_size=768

## Run
### supervised-EntitySimCSE-residence
> python fed_trainer.py --config configs/supervised/EntitySimCSE/residence/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/residence/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/residence/EntitySimCSE_L_12_E_4_H_768.yaml

### supervised-SimCSE-residence
> python fed_trainer.py --config configs/supervised/SimCSE/residence/SimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/supervised/SimCSE/residence/SimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/supervised/SimCSE/residence/SimCSE_L_12_E_4_H_768.yaml

### supervised-EntitySimCSE-institution
> python fed_trainer.py --config configs/supervised/EntitySimCSE/institution/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/institution/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/institution/EntitySimCSE_L_12_E_4_H_768.yaml

### supervised-SimCSE-institution
> python fed_trainer.py --config configs/supervised/SimCSE/institution/SimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/supervised/SimCSE/institution/SimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/supervised/SimCSE/institution/SimCSE_L_12_E_4_H_768.yaml

### supervised-EntitySimCSE-federated
> python fed_trainer.py --config configs/supervised/EntitySimCSE/federated/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/federated/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/supervised/EntitySimCSE/federated/EntitySimCSE_L_12_E_4_H_768.yaml

---
### unsupervised-EntitySimCSE-residence
> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/residence/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/residence/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/residence/EntitySimCSE_L_12_E_4_H_768.yaml

### unsupervised-SimCSE-residence
> python fed_trainer.py --config configs/unsupervised/SimCSE/residence/SimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/unsupervised/SimCSE/residence/SimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/unsupervised/SimCSE/residence/SimCSE_L_12_E_4_H_768.yaml

### unsupervised-EntitySimCSE-institution
> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/institution/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/institution/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/institution/EntitySimCSE_L_12_E_4_H_768.yaml

### unsupervised-SimCSE-institution
> python fed_trainer.py --config configs/unsupervised/SimCSE/institution/SimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/unsupervised/SimCSE/institution/SimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/unsupervised/SimCSE/institution/SimCSE_L_12_E_4_H_768.yaml

### unsupervised-EntitySimCSE-federated
> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/federated/EntitySimCSE_L_4_E_1_H_256.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/federated/EntitySimCSE_L_8_E_3_H_512.yaml

> python fed_trainer.py --config configs/unsupervised/EntitySimCSE/federated/EntitySimCSE_L_12_E_4_H_768.yaml
