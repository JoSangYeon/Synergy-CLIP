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

#### text domain
