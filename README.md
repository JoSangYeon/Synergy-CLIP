# Synergy-CLIP

##  Synergy-CLIP : Extending CLIP with Multi-Modal Integration for Robust Representation Learning

### Abstrat

![Intro](https://github.com/JoSangYeon/ECAI2024-Synergy-CLIP/assets/28241676/ae75a832-474d-4a5c-9ce5-47428e18aa97)

This paper presents "Synergy-CLIP", an innovative extension of the CLIP framework, enhancing the integration of vision, text, and audio modalities to improve multi-modal representation learning. Traditional approaches in artificial intelligence have often focused on binary modal combinations such as image-text; however, our framework addresses the challenge of integrating three modalities simultaneously, thus mimicking more complex human cognitive processes. We introduce a novel methodology for the Missing Modality Reconstruction(MMR) task, which leverages the robust representations derived from our tri-modal dataset to reconstruct any missing modality effectively. By employing a combination of vision-text-audio datasets and advanced learning algorithms, "Synergy-CLIP" enables not only superior performance in modality-specific downstream tasks such as zero-shot classification but also provides significant insights into the potential of comprehensive modal integration. Our empirical evaluations demonstrate the framework's effectiveness in enhancing model adaptability and robustness, thereby setting a new standard in multi-modal learning research. This advancement underscores the importance of expanding AI's capability to process and understand multi-modal data in a manner akin to human sensory and cognitive systems.

### Architecture

#### Synergy-CLIP

![step 1_3 - Synergy-CLIP](https://github.com/JoSangYeon/ECAI2024-Synergy-CLIP/assets/28241676/b6f812d9-121a-4045-a674-5e85af31385c)

### Result

#### Image domain
| Model               | Oxford-IIIT Pets | Flowers-102 | CIFAR-10 | CIFAR-100 |
|---------------------|------------------|-------------|----------|-----------|
| ResNet 152 [11]      | 93.00            | 89.60       | 93.50    | 78.00     |
| BiT-M [42]           | 92.40            | 99.30       | 97.60    | 88.20     |
| ViT-B [39]           | 90.40            | 97.60       | 96.70    | 86.30     |
| ViT-L [39]           | 92.90            | 98.70       | 97.90    | 89.00     |
| CLIP: ViT-B [11]     | 90.00            | 96.90       | 95.10    | 80.50     |
| CLIP: ViT-L [11]     | 95.10            | 99.20       | 98.00    | 87.50     |
| **Synergy-CLIP**     |                  |             |          |           |
| Baseₚ : img enc.     | 91.99 ± 0.05     | 97.68 ± 0.02| 97.31 ± 0.01 | 83.28 ± 0.04 |
| Basec : img enc.     | 91.76 ± 0.01     | 97.22 ± 0.04| 97.04 ± 0.01 | 87.25 ± 0.03 |
| Largeₚ : img enc.    | 94.40 ± 0.02     | 99.54 ± 0.01| 98.22 ± 0.01 | 90.82 ± 0.02 |
| Largec : img enc.    | 94.73 ± 0.01     | 98.80 ± 0.08| 98.39 ± 0.02 | 90.74 ± 0.02 |

#### Audio domain
| Model                   | ESC-50         | UrbanSound8k  |
|-------------------------|----------------|---------------|
| Human                   | 81.50          | -             |
| ESResNeXt [51]           | 95.20          | 89.14         |
| AST [40]                 | 95.60          | -             |
| AudioCLIP [14]           | 97.15          | 90.07         |
| CLAP [12]                | 96.70          | 87.96         |
| **Synergy-CLIP**         |                |               |
| Baseₚ : aud enc.         | 95.05 ± 0.02 (*97.75) | 89.31 ± 0.03 (*94.44) |
| Basec : aud enc.         | 94.55 ± 0.01 (*96.75) | 89.18 ± 0.02 (*94.55) |
| Largeₚ : aud enc.        | 94.70 ± 0.02 (*96.75) | 87.83 ± 0.04 (*92.64) |
| Largec : aud enc.        | 95.10 ± 0.02 (*96.75) | 88.50 ± 0.03 (*91.45) |

#### Text domain
| Model                      | MNLI-M | MNLI-MM | QNLI  | QQP   | SST-2 | CoLA  | MRPC  | RTE   | avg   |
|----------------------------|--------|---------|-------|-------|-------|-------|-------|-------|-------|
| BiLSTM+Attn, ELMo           | 72.40  | 72.40   | 75.20 | 83.60 | 91.50 | 44.10 | 82.10 | 52.70 | 71.75 |
| Bert-base [44]              | 84.00  | 84.20   | 91.00 | 87.60 | 91.60 | 52.10 | 90.20 | 69.50 | 82.43 |
| RoBERTa [41]                | 90.20  | 90.20   | 94.70 | 92.20 | 96.40 | 68.00 | 90.90 | 86.60 | 88.65 |
| CLIP: Text enc [31]         | 71.80  | 72.50   | 73.00 | 82.70 | 86.20 | 66.00 | 81.40 | 53.80 | 66.00 |
| **Synergy-CLIP**            |        |         |       |       |       |       |       |       |       |
| Baseₚ : txt enc.            | 87.31  | 87.04   | 91.60 | 90.10 | 93.12 | 69.12 | 90.09 | 61.73 | 83.76 |
| Basec : txt enc.            | 87.53  | 87.31   | 91.93 | 90.11 | 93.35 | 69.13 | 90.39 | 65.70 | 84.30 |
| Largeₚ : txt enc.           | 90.23  | 90.39   | 94.45 | 91.74 | 96.10 | 83.89 | 90.56 | 74.01 | 88.80 |
| Largec : txt enc.           | 90.29  | 90.49   | 94.75 | 91.71 | 96.22 | 85.23 | 90.80 | 79.42 | 89.86 |


### Run code
```python
CUDA_VISIBLE_DEVICES=0,1 python main_pretraining.py --SEED 77 --WORLD_SIZE 2 --PORT 12345 --IS_BASE True --IS_CAPTIONED True --learning_rate 5e-6
```
