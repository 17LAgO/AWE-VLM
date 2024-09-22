# AWE-VLM
For study VLM with new motion

 <p align="center">
</p>
<details> 
<summary>ℹ️ <i>通用VLM Architectures </i></summary>   
# 👁️‍🗨️Awesome VLM Architectures [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![VLM](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/5c9ee091-1f37-4d92-8398-a7d4e006c014)

**Vision-Language Models (VLMs)** feature a multimodal architecture that processes image and text data simultaneously. They can perform **Visual Question Answering (VQA)**, **image captioning** and **Text-To-Image search** kind of tasks. VLMs utilize techniques like multimodal fusing with cross-attention, masked-language modeling, and image-text matching to relate visual semantics to textual representations. This repository contains information on famous Vision Language Models (VLMs), including details about their architectures, training procedures, and the datasets used for training. **Click to expand for further details for every architecture**
- 📙 <a href="https://github.com/gokayfem/ComfyUI_VLM_nodes">Visit my other repo to try Vision Language Models on ComfyUI</a>

## Contents

- [Architectures](#architectures)
- [Important References](#important-references)

## Architectures

### **LLaVA: Large Language and Vision Assistant - Visual Instruction Tuning**

LLaVA seamlessly integrates a pre-trained language model (Vicuna) with a visual encoder (CLIP) using a simple linear layer, creating a robust architecture capable of effectively processing and understanding language-image instructions.

[![arXiv](https://img.shields.io/badge/arXiv-2304.08485-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.08485) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/haotian-liu/LLaVA) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://llava.hliu.cc/)  
Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/722f0fbb-ea52-4a8a-ab1e-bec45ca7d04f" />
</p>
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**LLaVA**: At the heart of LLaVA's architecture is the fusion of a pre-trained language model with a visual model, specifically designed to process and understand language-image instruction data effectively. This integration enables LLaVA to leverage the distinct strengths of both models, employing the CLIP visual encoder for robust image feature extraction and the Vicuna language model for intricate language instruction processing. A noteworthy feature of this architecture is the use of **a simple linear layer** that bridges image features to the word embedding space, facilitating a seamless alignment between visual and linguistic representations. The training methodology of LLaVA is meticulously structured into a two-stage instruction-tuning procedure. Initially, the model undergoes pre-training focused on feature alignment, utilizing a carefully filtered dataset to synchronize image features with LLM word embeddings. Subsequently, the model is fine-tuned end-to-end on tailored tasks such as multimodal chatbot functionalities and Science QA, with the aim of refining its instruction-following prowess. This sophisticated training regimen is underpinned by the use of multimodal instruction-following data generated via GPT-4, converting image-text pairs into formats conducive to instruction-following tasks. The alignment of text and image data is innovatively achieved through **a trainable projection matrix**, converting visual features into language embedding tokens within a unified dimensional space, thereby enhancing the model's ability to encode vision and text cohesively.The datasets deployed for LLaVA's training and evaluation are strategically selected to bolster its multimodal capabilities. The Filtered CC3M dataset serves as the foundation for pre-training, aligning visual and language features, while the LLaVA-Instruct-158K dataset generated using GPT-4 is pivotal for fine-tuning the model on diverse multimodal tasks. Additionally, the ScienceQA dataset plays a critical role in assessing LLaVA's proficiency in multimodal reasoning tasks, demonstrating the model's comprehensive training and its potential to significantly advance the field of multimodal interaction and understanding.
</details> 

### **LLaVA 1.5: Improved Baselines with Visual Instruction Tuning**

LLaVA 1.5 enhances its multimodal understanding by replacing its initial linear projection with a more powerful multi-layer perceptron (MLP), enabling a deeper integration of visual features from CLIP-ViT-L-336px and linguistic data.

[![arXiv](https://img.shields.io/badge/arXiv-2310.03744-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.03744)  
Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c7112b75-3b86-48a2-9c0f-f1dc1dc6ee06" />
</p>
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA 1.5**: This iteration introduces a refined architecture that incorporates a CLIP-ViT-L-336px vision encoder alongside **a multi-layer perceptron (MLP) projection layer**. This combination not only boosts the model's data efficiency but also its performance across various benchmarks, showcasing a leap in multimodal understanding. The architecture's core components, the CLIP-ViT-L for visual encoding and the MLP for vision-language cross-modal connection, work synergistically to enhance the model's capacity to integrate and interpret visual and linguistic inputs.Training methods have been optimized in LLaVA 1.5 to achieve unprecedented performance on 11 benchmarks, utilizing a two-stage approach that emphasizes efficient feature alignment and fine-tuning with VQA data specifically tailored for academic tasks. The paper highlights a shift towards more sophisticated multimodal alignment techniques, **replacing the original linear projection** with a more powerful **MLP vision-language connector**. This strategic improvement facilitates a deeper and more nuanced integration of visual and linguistic data. Moreover, the adoption of an MLP-based vision-language connector for alignment fusion methods further strengthens the model's ability to merge visual and textual representations effectively, ensuring closer alignment in the embedding space.The utilization of datasets such as VQA-v2, GQA, and other academic-task-oriented VQA datasets, enriched with OCR and region-level perception data, underscores the model's enhanced visual understanding and reasoning capabilities. These datasets play a crucial role in elevating LLaVA 1.5's performance, enabling it to set new standards with academic-task-oriented data. Through these advancements, LLaVA 1.5 not only pushes the boundaries of multimodal learning but also sets a new benchmark for future research in the field.
</details> 

### **LLaVA 1.6: LLaVA-NeXT Improved reasoning, OCR, and world knowledge**

LLaVA-NeXT advances on LLaVA-1.5 by incorporating high-resolution image processing, enhancing visual reasoning and OCR capabilities, while maintaining a data-efficient design through knowledge transfer from its predecessor and a refined training process.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://llava-vl.github.io/blog/2024-01-30-llava-next/)  
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/032ef144-ec10-41da-80a1-2cecd66c86ee" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA-NeXT**: Represents a significant step forward in the evolution of large language models with visual capabilities, building upon the foundations laid by LLaVA-1.5. This model introduces several enhancements aimed at improving image resolution, visual reasoning, optical character recognition (OCR), and the integration of world knowledge, all while retaining the minimalist and data-efficient design of its predecessor. The architecture of LLaVA-NeXT is optimized for high performance, supporting input image resolutions up to 672x672, 336x1344, and 1344x336 pixels. This improvement facilitates a more detailed visual perception, which, coupled with an enhanced visual instruction tuning data mixture, significantly bolsters the model's reasoning and OCR capabilities. Furthermore, LLaVA-NeXT achieves efficient deployment through the use of SGLang, a feature that underscores its design's focus on performance and data efficiency.Training LLaVA-NeXT requires less than 1 million visual instruction tuning samples, leveraging the **pre-trained connector** from LLaVA-1.5 for efficient knowledge transfer. The training process, remarkably swift, utilizes 32 A100 GPUs and completes in approximately one day, a testament to the model's efficient design and deployment strategy. The alignment techniques in LLaVA-NeXT are particularly noteworthy, utilizing high-resolution images and a high-quality data mixture to enhance the model's capabilities in visual conversation and instruction following. The model's use of dynamic high-resolution techniques, known as 'AnyRes', allows for effective handling of images with varying resolutions, improving the model's overall visual understanding.The datasets employed in training LLaVA-NeXT, including LAION-GPT-V, ShareGPT-4V, DocVQA, SynDog-EN, ChartQA, DVQA, and AI2D, are meticulously chosen to augment the model's visual reasoning, OCR capabilities, and comprehension of charts and diagrams. This strategic selection aims to elevate the model's performance across a wide range of multimodal tasks, emphasizing its enhanced ability to process and understand complex visual information. Through these improvements, LLaVA-NeXT sets a new benchmark for models at the intersection of language and vision, offering unprecedented capabilities in visual reasoning, OCR, and the application of world knowledge in multimodal contexts.
</details> 

### **PaliGemma: A Versatile and Transferable 3B Vision-Language Model**

PaliGemma is a compact, open-source vision-language model designed to be easily transferable to a diverse range of tasks. It combines a powerful SigLIP image encoder with the Gemma-2B language model, achieving strong performance on over 40 diverse tasks, including standard VLM benchmarks, remote-sensing, and segmentation. PaliGemma is pretrained using a multi-stage approach, focusing on maximizing the density of learning signal and providing different checkpoints with varying image resolutions. This versatile foundation model is easily fine-tuned for specific tasks and serves as a valuable tool for researchers and practitioners exploring the capabilities of VLMs.

[![arXiv](https://img.shields.io/badge/arXiv-2407.07726-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2407.07726) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/big-vision/paligemma)  
Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, Xiaohua Zhai  

<p align="center">
<img src="https://github.com/user-attachments/assets/186371d0-6861-4b68-b32e-fee77cc75ef2" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

PaliGemma stands out as a highly versatile and transferable 3-billion parameter Vision-Language Model (VLM) meticulously designed for broad applicability across a wide spectrum of visual-language tasks. Its foundation lies in the integration of two powerful components: a SigLIP-So400m vision encoder, known for its exceptional performance despite its compact size, and the Gemma-2B language model, a pretrained autoregressive decoder-only model from the Gemma family. This combination enables PaliGemma to effectively process and understand both visual and textual information, making it adept at handling tasks ranging from image captioning and visual question answering to more specialized tasks like remote-sensing and segmentation. PaliGemma's architecture is streamlined and efficient. It uses a simple linear projection to align the visual features extracted by the SigLIP encoder with the vocabulary tokens of the Gemma language model, enabling seamless fusion of the two modalities. A key aspect of PaliGemma's training is the emphasis on "density of learning signal," prioritizing a broad range of skills and knowledge over achieving high zero-shot performance. This approach involves a multi-stage pretraining process that starts with unimodal pretraining of individual components using publicly available checkpoints, followed by extensive multimodal pretraining on a diverse mixture of large-scale vision-language tasks. Notably, PaliGemma deviates from the common practice of freezing the image encoder during pretraining, allowing it to learn spatial and relational understanding from complex tasks like captioning. To further enhance its capabilities, PaliGemma undergoes a resolution increase stage, where it is trained on higher-resolution images, enabling it to handle tasks that benefit from finer visual details. This multi-stage pretraining process results in a family of three PaliGemma checkpoints at varying image resolutions (224px, 448px, and 896px), each pretrained with broad visual knowledge. These checkpoints serve as strong base models that can be easily transferred to specific downstream tasks. PaliGemma's transferability is demonstrated through its impressive performance on over 30 academic benchmarks, including those involving multiple images, such as NLVR2 and short-video understanding tasks. The model's ability to adapt quickly to new tasks with minimal fine-tuning highlights its versatility and makes it a valuable tool for exploring and advancing the capabilities of VLMs. Furthermore, the model's open-source nature, along with its straightforward architecture and training recipe, encourages further research and experimentation within the VLM community, driving progress towards more powerful and general-purpose multimodal AI systems.
</details>

### **Idefics2**

IDEFICS2, an 8B parameter open-source vision-language model, efficiently processes interleaved image and text sequences by combining a SigLIP vision encoder, a Mistral-7B LLM, and a Perceiver pooling layer with MLP projection for robust text encoding, excelling in tasks like OCR and document understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2405.02246-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.02246) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/HuggingFaceM4/idefics-8b)  
Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/c197c8c5-8da2-4d96-8999-8e05e81f1506" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
IDEFICS2 is an 8B parameter open-source vision-language model adept at handling interleaved image and text sequences. IDEFICS2 utilizes a vision-language architecture designed for efficient processing of image and text sequences. It employs the SigLIP model as the vision encoder, extracting features from images in their native resolutions and aspect ratios. The Mistral-7B model serves as the LLM backbone, providing language understanding and generation capabilities. For text encoding, IDEFICS2 leverages a **Perceiver pooling layer** followed by an **MLP projection** to integrate visual features with the LLM's embedding space. This combination of vision encoder, LLM, and text encoder enables IDEFICS2 to handle various multimodal tasks, with a particular focus on OCR and document understanding.  The model is trained on a diverse dataset encompassing OBELICS, LAION Coco, and PMD, with additional data for OCR tasks. Fine-tuning is performed on instruction datasets like The Cauldron and OpenHermes-2.5.
</details> 

### **Idefics3-8B: Building and Better Understanding Vision-Language Models**

Idefics3-8B is a powerful open-source vision-language model (VLM) that significantly outperforms its predecessor, Idefics2-8B, while being trained efficiently and exclusively on open datasets. It leverages a straightforward pipeline and introduces Docmatix, a massive dataset for document understanding, to achieve state-of-the-art performance within its size category across various multimodal benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2408.12637-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.12637) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/HuggingFaceM4/idefics3)  
Hugo Laurençon, Andrés Marafioti, Victor Sanh, Léo Tronchon  
<p align="center">
<img src="https://github.com/user-attachments/assets/5e61fec2-b41b-4ad8-a167-1966f169b866" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Idefics3-8B builds upon the foundation of pre-trained unimodal models, specifically Llama 3.1 instruct as the language model and SigLIP-SO400M as the vision encoder. It adopts a self-attention architecture, where visual features are treated as tokens and concatenated with text tokens before being fed into the LLM. To enhance OCR capabilities and address the bottleneck of limited visual tokens per image, Idefics3-8B replaces the perceiver resampler used in Idefics2 with a simple pixel shuffle strategy, similar to InternVL-1.5. This strategy reduces the number of image hidden states by a factor of 4, allowing for the encoding of larger images (up to 364x364 pixels) into 169 visual tokens. The model utilizes an image-splitting strategy during both training and inference, dividing the original image into a matrix of 364x364 pixel tiles. To preserve the 2D structure and positional information of these tiles, a text token '\n' is inserted after each row of tiles, and the downscaled original image is appended to the sequence. Additionally, each tile is prepended with textual tokens indicating its position in the matrix. The training process consists of three stages of pre-training followed by supervised fine-tuning. In the first pre-training stage, the backbones (LLM and vision encoder) are frozen, and only the newly initialized parameters are trained. The maximum image resolution is gradually increased from 364² to 1820². From the second stage onward, the backbones are efficiently trained using DoRA (a variant of LoRA), and larger images are introduced into the training data. The final pre-training stage focuses on training with large synthetic datasets, including Docmatix, Websight, LNQA, PixelProse, and ChartGemma. During supervised fine-tuning, NEFTune noise is applied to the inputs, and the loss is calculated only on the answer tokens. The learning rate is kept constant for the first two pre-training stages and linearly decayed to zero during the final pre-training stage and supervised fine-tuning. Idefics3-8B demonstrates significant improvements over Idefics2, particularly in document understanding tasks, achieving a 13.7-point improvement on DocVQA. This highlights the effectiveness of the Docmatix dataset and the architectural choices made in Idefics3-8B. The model also achieves state-of-the-art performance within its size category across various multimodal benchmarks, including MMMU, MathVista, MMStar, and TextVQA, showcasing its strong capabilities in visual understanding and reasoning.
</details>

### **InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model**

InternLM-XComposer2 excels in free-form text-image composition and comprehension by connecting a CLIP pre-trained vision encoder with the powerful InternLM-2 LLM using a novel Partial LoRA module, enabling efficient alignment of visual and language tokens for enhanced multimodal understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2401.16420-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.16420) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/InternLM/InternLM-XComposer) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Willow123/InternLM-XComposer)  
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/732d3b7b-02de-42d3-ae76-800bf035b391" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**InternLM-XComposer2**: This model introduces a sophisticated architecture that leverages a vision encoder and a Large Language Model (LLM), interconnected through a Partial Low-Rank Adaptation (LoRA) module. This innovative setup allows InternLM-XComposer2 to effectively process both images and text, employing visual tokens generated by the vision encoder alongside language tokens derived from the tokenized text. The vision encoder, pre-trained using CLIP for image-language contrastive learning, and InternLM-2, which serves as the LLM with multi-lingual capabilities, are key components of this architecture. **The Partial LoRA** module distinguishes itself by aligning visual and language tokens through low-rank adaptation applied specifically to visual tokens, enhancing the model's multimodal understanding and processing efficiency. The training methodology of InternLM-XComposer2 is multifaceted, focusing on fine-tuning the vision encoder and Partial LoRA to align visual tokens with the LLM across various datasets. This process involves general semantic alignment, world knowledge alignment, and vision capability enhancement to refine the model's ability to interpret image information and compose text-image content. Supervised fine-tuning further includes multi-task training and free-form text-image composition, aiming to optimize the model's performance in leveraging image information for comprehensive text-image generation and understanding. Alignment techniques and fusion methods in InternLM-XComposer2 utilize the Partial LoRA module for the effective integration of different modalities, thereby enriching the LLM with modality-specific knowledge while preserving its inherent capabilities. This selective enhancement of visual tokens through Partial LoRA enables the model to exhibit robust performance across visual and textual domains, facilitating detailed perception, logical reasoning, and extensive knowledge integration in multimodal understanding. The model employs a diverse array of datasets, including ShareGPT4V-PT, COCO, Nocaps, TextCaps, and many others, for pre-training and supervised fine-tuning. These datasets serve to equip InternLM-XComposer2 with a broad range of capabilities, including general semantic alignment, world knowledge alignment, vision capability enhancement, and the facilitation of free-form text-image composition, marking a significant advancement in the field of vision-language large models.
</details> 

### **InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD**  

InternLM-XComposer2-4KHD, building on its predecessor, pioneers high-resolution image handling in LVLMs by employing dynamic resolution with automatic patch configuration, adapting to resolutions from 336 pixels up to 4K HD for enhanced visual understanding without distortion.

[![arXiv](https://img.shields.io/badge/arXiv-2404.06512v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2404.06512v1)  
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/c09b67fb-32eb-43de-82fa-96c3af22caf4" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>    
  
**InternLM-XComposer2-4KHD**: Cutting-edge Large Vision-Language Model (LVLM) designed to handle ultra-high resolutions, up to 4K HD and beyond, while also supporting diverse resolutions from 336 pixels. The model builds upon the InternLM-XComposer2 architecture, incorporating a novel **dynamic resolution with automatic patch configuration** technique. This allows the model to dynamically adjust patch layouts and counts based on the input image's aspect ratio, enabling efficient processing of high-resolution images while preserving their original proportions. To address potential ambiguity arising from variable patch configurations, a newline token is introduced to delineate rows of patch tokens, significantly improving performance. InternLM-XComposer2-4KHD is pre-trained on a diverse dataset, including image-caption pairs, concept knowledge, and OCR datasets, focusing on enhancing high-resolution and structural image understanding. Supervised fine-tuning further incorporates a mixed-resolution strategy, employing higher resolution for tasks requiring fine-grained detail, like HD-OCR tasks, and dynamically adjusted resolution for other tasks. This approach enables the model to excel in both high-resolution scenarios and general vision-language understanding tasks.
</details> 

### **InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output**

InternLM-XComposer-2.5 (IXC-2.5) is a versatile Large Vision Language Model (LVLM) designed to handle long-contextual input and output, excelling in various text-image comprehension and composition tasks. It achieves performance comparable to GPT-4V with a significantly smaller 7B LLM backend, demonstrating its efficiency and scalability.

[![arXiv](https://img.shields.io/badge/arXiv-2407.03320-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2407.03320) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/InternLM/InternLM-XComposer) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/Willow123/InternLM-XComposer)  
Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang  

<p align="center">
<img src="https://github.com/user-attachments/assets/1330a013-930b-4b23-90dc-94616b59ca0b" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

InternLM-XComposer-2.5 builds upon its previous iterations (IXC-2 and IXC-2-4KHD) and features a three-component architecture: a lightweight **OpenAI ViT-L/14 vision encoder**, a powerful InternLM2-7B LLM, and **Partial LoRA** for efficient alignment between the visual and language modalities. IXC-2.5 supports diverse input modalities, including text, single/multiple images, and videos. It utilizes a Unified Dynamic Image Partition strategy to handle high-resolution images and videos, resizing and padding them into smaller patches while preserving aspect ratios. For videos, frames are sampled and concatenated along the short side, creating a high-resolution composite image. The model is pre-trained in three stages: general semantic alignment, world knowledge alignment, and vision capability enhancement, using a diverse range of datasets. During pre-training, the LLM is frozen, and the vision encoder and Partial LoRA are fine-tuned to align visual tokens with the LLM. Supervised fine-tuning is then performed on a collection of datasets covering various tasks, including captioning, visual question answering, multi-turn QA, science QA, chart QA, math QA, OCR QA, video understanding, and conversation. This fine-tuning process involves jointly training all components with a weighted data sampling strategy and specific learning rate schedules for each component. IXC-2.5 also introduces two novel applications: crafting webpages and composing high-quality text-image articles. For webpage generation, the model is trained on a combination of synthetic and real-world web data, enabling it to generate HTML, CSS, and JavaScript code based on screenshots, instructions, or resume documents. For article composing, IXC-2.5 leverages Chain-of-Thought (CoT) and Direct Preference Optimization (DPO) techniques to enhance the quality of written content. This involves rewriting original prompts using CoT, generating diverse responses using different random seeds, and training a reward model to select preferred responses, ultimately leading to more creative and high-quality article generation.
</details>

### **DeepSeek-VL: Towards Real-World Vision-Language Understanding**  

DeepSeek-VL, utilizing a hybrid vision encoder combining SigLIP-L and SAM-B, excels in real-world vision-language understanding by efficiently processing high-resolution images and integrating extracted features with a DeepSeek LLM backbone through a two-layer hybrid MLP adapter.

[![arXiv](https://img.shields.io/badge/arXiv-2401.16420-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2403.05525) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/deepseek-ai/DeepSeek-VL) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/deepseek-ai/DeepSeek-VL-7B)  
Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, Chong Ruan  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/7b7283d2-b2d5-4ab6-891a-18a9760ef7ca" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  

**DeepSeek-VL**: Employs a hybrid vision encoder architecture, fusing a **SigLIP-L encoder** for semantic understanding with a **SAM-B encoder** for high-resolution detail extraction. This allows for efficient processing of 1024x1024 images while capturing both global and fine-grained visual features. **A two-layer hybrid MLP adapter** then integrates these features with the DeepSeek LLM backbone. The model is pre-trained on a diverse dataset encompassing web screenshots, PDFs, OCR, charts, and knowledge-based content from sources like Common Crawl, Web Code, E-books, and arXiv articles. This pretraining is further refined using a curated instruction-tuning dataset based on real user scenarios and categorized into a comprehensive taxonomy covering recognition, conversion, analysis, reasoning, evaluation, and safety tasks. By combining this diverse data with its unique architecture and fusion strategies, DeepSeek-VL aims to deliver robust performance across a wide range of real-world vision-language applications.  
</details> 

### **MANTIS: Mastering Multi-Image Understanding Through Interleaved Instruction Tuning** 

MANTIS is a family of open-source large multimodal models that demonstrate state-of-the-art performance on multi-image visual language tasks. By focusing on instruction tuning with a carefully curated multi-image dataset, MANTIS achieves superior results using significantly less data than models trained with massive web datasets. This efficient approach opens new avenues for developing powerful multi-image LMMs with limited resources.

[![arXiv](https://img.shields.io/badge/arXiv-2405.01483-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.01483) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/TIGER-AI-Lab/Mantis) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TIGER-Lab/Mantis)  
Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, Wenhu Chen  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/dd4bbdf4-5ab9-4e12-89bd-94c5beb2d114" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary> 

**Mantis**: a powerful and efficient multi-image Large Multimodal Models (LMMs), demonstrating that massive pre-training on noisy web data is not the only path towards achieving state-of-the-art performance in complex visual-language tasks. Instead, MANTIS focuses on instruction tuning using high-quality, academic-level data, achieving remarkable results on various multi-image benchmarks while using significantly less data than its counterparts. Central to MANTIS's success is the meticulously curated MANTIS-INSTRUCT dataset, a collection of 721K multi-image instruction data carefully designed to instill four crucial skills: co-reference, comparison, reasoning, and temporal understanding. These skills equip MANTIS with a comprehensive toolkit for tackling the challenges of multi-image understanding. Co-reference enables the model to understand references like "second image" in natural language and correctly identify the corresponding image within the input. Comparison fosters the ability to analyze and identify subtle differences and commonalities between multiple images, a skill crucial for tasks like visual similarity assessment and difference description. Reasoning empowers the model to go beyond simple comparisons and make complex inferences by combining its world knowledge with the information extracted from multiple images, allowing it to solve intricate logical reasoning puzzles and answer challenging multi-image questions. Finally, temporal understanding equips MANTIS with the capability to process and understand image sequences, capturing the dynamic aspects of videos, comics, and other temporal visual data. MANTIS leverages a simple yet effective architecture based on existing pre-trained LLMs like LLaMA-3 and vision transformer encoders from CLIP or SigLIP. A multimodal projector, similar to the one used in LLaVA, aligns the visual embeddings with the text embeddings, facilitating their seamless integration within the LLM. This streamlined approach avoids the complexity of previous architectures like Q-Former while retaining high performance. Extensive evaluations on five multi-image benchmarks, including NLVR2, QBench, BLINK, MVBench, and a newly curated Mantis-Eval dataset, demonstrate MANTIS's superior performance, exceeding existing open-source LMMs and even matching the results of the powerful GPT-4V. Notably, MANTIS surpasses Idefics2-8B, a model pre-trained on 200x larger interleaved multi-image data, showcasing the effectiveness of instruction tuning with high-quality academic-level data. Furthermore, MANTIS retains strong single-image performance on par with existing state-of-the-art models, demonstrating its versatility and adaptability. MANTIS's impressive results, combined with its efficient training and open-source nature, offer a compelling alternative to traditional pre-training-heavy approaches, opening new possibilities for researchers and practitioners seeking to develop powerful and versatile multi-image LMMs with minimal computational resources.
</details> 

### **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**

Qwen-VL distinguishes itself by integrating a Vision Transformer with a large language model through a novel vision-language adapter, employing cross-attention mechanisms for precise alignment of visual and linguistic data, achieving high performance in various vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2308.12966-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.12966) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/qwenlm/qwen-vl) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Qwen/Qwen-VL-Plus)  
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c9358aad-63e2-44d3-b3af-38e9d4f6aeaa" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Qwen-VL**: Represents an advanced architecture in the vision-language domain, constructed on a foundational large language model with the integration of a Vision Transformer (ViT) for visual encoding. This model stands out for its innovative approach to processing and aligning visual and linguistic data, featuring a **vision-language adapter equipped with cross-attention mechanisms**. These mechanisms enable the efficient compression and integration of image features into the language model, a critical component for achieving precise alignment between visual inputs and text. The architecture's design focuses on optimizing the handling of image features, employing a position-aware strategy to maintain spatial relevance of visual data when merged with textual information.The training methodology of Qwen-VL is meticulously structured into **three distinct phases**, starting with an **initial pre-training** on a diverse collection of weakly labeled image-text pairs. This is followed by **multi-task pre-training**, utilizing high-quality annotated datasets and larger input resolutions to refine the model's capabilities in various tasks such as instruction following and dialogue. The final phase involves **supervised fine-tuning**, further honing the model's performance across a spectrum of vision-language tasks. Special tokens and bounding box inputs are utilized for differentiating between image and text inputs and achieving fine-grained visual understanding, respectively.Qwen-VL's alignment techniques are innovative, employing a cross-attention mechanism within its vision-language adapter to fuse visual and textual features effectively. This approach ensures the preservation of spatial information post feature compression through the use of positional encodings. The model leverages an extensive suite of datasets for training, including LAION-en, LAION-zh, and various others for pre-training, alongside specialized datasets like GQA, VGQA, and VQAv2 for multi-task pre-training. These datasets are instrumental in supporting a broad array of vision-language tasks, emphasizing multilingual capabilities, fine-grained visual understanding, and the model's proficiency in captioning, visual question answering, grounding, and OCR tasks.
</details> 

### **Qwen2-VL: A Powerful Open-Source Vision-Language Model for Image and Video Understanding**

Qwen2-VL is the latest iteration of the Qwen vision-language model family, building upon the Qwen-VL architecture and introducing significant enhancements for improved understanding of images and videos. It excels in various tasks, including visual question answering, dialogue, content creation, and even agent-based control of devices like mobile phones and robots.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen2-VL) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)  
Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren  

<p align="center">
<img src="https://github.com/user-attachments/assets/37c2fb7a-66e1-475f-86e4-f00b4ac1c879" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Qwen2-VL continues to leverage the core architecture of Qwen-VL, combining a Vision Transformer (ViT) with approximately 600M parameters and Qwen2 language models. This ViT is designed to handle both image and video inputs seamlessly. The key architectural improvements in Qwen2-VL include Naive Dynamic Resolution support and Multimodal Rotary Position Embedding (M-ROPE). Naive Dynamic Resolution allows the model to handle arbitrary image resolutions by mapping them into a dynamic number of visual tokens. This ensures that the model input accurately reflects the information content of the image, regardless of its original resolution. This approach is more aligned with human visual perception, which adapts to different image sizes and resolutions. M-ROPE enhances the model's ability to capture positional information in multimodal inputs. It deconstructs the original rotary embedding into three parts, representing temporal, height, and width information. This allows the LLM to simultaneously process and integrate 1D textual, 2D visual (image), and 3D video positional information, leading to a more comprehensive understanding of the input sequence. These architectural enhancements, combined with a robust training process, enable Qwen2-VL to achieve state-of-the-art performance on various visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, and MTVQA. It can also understand videos over 20 minutes long, enabling high-quality video-based question answering, dialogue, and content creation. Furthermore, Qwen2-VL's capabilities in complex reasoning and decision-making allow it to be integrated with devices like mobile phones and robots for automatic operation based on visual input and text instructions. The model also supports multilingual understanding of text within images, including most European languages, Japanese, Korean, Arabic, and Vietnamese, broadening its applicability to a global user base.
</details>

### **moondream1 and moondream2**

moondream1 and moondream2 are vision-language models with moondream2 building upon moondream1's SigLIP vision encoder and Phi-1.5 language backbone by incorporating an MLP projector for enhanced visual and textual representation alignment.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vikhyat/moondream) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vikhyatk/moondream2)  
@vikhyatk
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/e979d327-3423-4a91-92f2-02a3dc3189a8" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
  
**moondream1 and moondream2**: A series of vision-language models. moondream1 is a 1.6B parameter model that leverages **SigLIP** as the vision encoder and **Phi-1.5** as the language backbone, trained on the LLaVA dataset. moondream2 expands upon this foundation, utilizing a 1.86B parameter model initialized with weights from SigLIP and Phi-1.5. It incorporates **an MLP projector** to bridge the visual and textual representations, potentially leading to enhanced vision-language alignment and improved performance across various tasks.
</details>

### **SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models**

SPHINX-X refines multi-modal large language models by streamlining its architecture to use two visual encoders, CLIP-ConvNeXt and DINOv2, and implementing an efficient single-stage training process for enhanced performance across diverse multi-modal tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2402.05935-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2402.05935) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alpha-vllm/llama2-accessory) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/Alpha-VLLM/SPHINX)  
Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, Wenqi Shao, Chao Xu, Conghui He, Junjun He, Hao Shao, Pan Lu, Hongsheng Li, Yu Qiao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/1c4e9a86-9a21-4911-bcb6-d2a79c181510" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**SPHINX-X**: Represents an advanced iteration in the development of Multi-modal Large Language Models (MLLM), building upon its predecessor, SPHINX, by optimizing both architecture and training efficiency. The core modifications introduced in SPHINX-X include the elimination of redundant visual encoders, the incorporation of **learnable skip tokens** to bypass **fully-padded sub-images**, and the simplification of the multi-stage training process into a singular, **all-in-one training** paradigm. This approach is designed to enhance the model's efficiency and effectiveness across a broad spectrum of multi-modal tasks. The architecture of SPHINX-X retains two key visual encoders, **CLIP-ConvNeXt and DINOv2**, ensuring robust text-image alignment capabilities, especially for high-resolution images and varied aspect ratios. This streamlined model architecture enables a unified encoding approach for both vision and text, emphasizing scalable and efficient training methodologies. The training strategy is comprehensive, directly engaging all model parameters across a wide-ranging multi-modal dataset, which encompasses public resources covering language, vision, and vision-language tasks. Additionally, SPHINX-X enriches this dataset with specially curated OCR-intensive and Set-of-Mark datasets to further extend the model's versatility and generalization capabilities. The datasets utilized in SPHINX-X aim to foster a deep, comprehensive understanding across multiple domains, enhancing the model's performance in OCR, document layout detection, and fine-grained multi-modal understanding. By training over various base Large Language Models (LLMs) with different parameter sizes and multilingual capabilities, SPHINX-X achieves a spectrum of MLLMs that showcase a strong correlation between multi-modal performance and the scales of data and parameters involved. This strategy allows SPHINX-X to set a new benchmark in multi-modal large language model performance, significantly advancing the field's capabilities in handling complex, multi-domain tasks.
</details>

### **BLIP: Bootstrapping Language-Image Pre-training**

BLIP introduces a versatile Multimodal Mixture of Encoder-Decoder (MED) architecture, integrating a visual transformer and a BERT-based text encoder with cross-attention layers, enabling unified vision-language understanding and generation across a wide range of tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2201.12086) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/BLIP)  
Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/27db1037-2b48-4097-9891-019ba77fc536" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIP**: Introduces an innovative approach to unified vision-language understanding and generation through its Multimodal Mixture of Encoder-Decoder (MED) architecture. This architecture is designed to be highly versatile, capable of serving as a unimodal encoder, an image-grounded text encoder, or an image-grounded text decoder. This flexibility allows BLIP to adeptly handle a wide array of vision-language tasks, showcasing its adaptability across various applications. The MED architecture incorporates a Visual Transformer to encode images, a BERT-based text encoder for processing textual information, additional **cross-attention layers** to facilitate image-text interaction, and **causal self-attention layers** for generating text based on image inputs. These components enable BLIP to support three key functionalities: encoding of either modality on its own, encoding of text grounded in images, and decoding of text from images, thus covering a comprehensive range of tasks from understanding to generation.BLIP's training methodology is grounded in the joint optimization of three pre-training objectives: Image-Text Contrastive Learning (ITC), Image-Text Matching (ITM), and Image-Conditioned Language Modeling (LM). These objectives are designed to align visual and textual features, learn fine-grained image-text alignment, and enable text generation from images, respectively. The model utilizes a mix of human-annotated and web-collected noisy image-text pairs for training, balancing the precision of manually annotated data with the scale and diversity of data collected from the web. This approach ensures robustness and scalability in BLIP's performance across vision-language tasks.For alignment and fusion of multimodal information, BLIP employs ITC and ITM losses to achieve precise text-image alignment, utilizing a multimodal representation that accurately captures the nuanced relationship between visual and textual data. The architecture's cross-attention layers play a crucial role in incorporating visual information into the text encoder for image-grounded text encoding. Simultaneously, modifications to the self-attention layers in the decoder facilitate text generation, effectively merging vision and text for unified processing. BLIP's pre-training leverages a diverse set of datasets, including COCO, Visual Genome, Conceptual Captions, Conceptual 12M, SBU Captions, and LAION. These datasets are instrumental in learning a broad spectrum of vision-language tasks, with high-quality human-annotated pairs and extensive web datasets providing the necessary depth and breadth for comprehensive pre-training.
</details> 

### **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

BLIP-2 leverages the power of frozen pre-trained image encoders and large language models, connecting them through a lightweight Querying Transformer (Q-Former) to efficiently extract and integrate visual features for enhanced vision-language understanding and generation.

[![arXiv](https://img.shields.io/badge/arXiv-2301.12597-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2301.12597) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Salesforce/BLIP2)  
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/604460f9-478c-4cc1-ba35-287447c04b26" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIP-2**: The model architecture integrates frozen pre-trained image encoders and large language models (LLMs), employing a lightweight **Querying Transformer (Q-Former)** to facilitate the interaction between these modalities. The Q-Former plays a crucial role in extracting and integrating visual features relevant to textual queries, allowing for a more nuanced understanding and generation of language based on visual inputs.The training methodology of BLIP-2 is structured around a two-stage pre-training strategy. Initially, it focuses on learning vision-language representations utilizing the frozen image encoders. Subsequently, it advances to vision-to-language generative learning, leveraging the capabilities of frozen LLMs. This strategy, coupled with the use of **learnable query vectors within the Q-Former**, enables effective vision-language alignment. The alignment process is further enhanced through fusion methods that extract language-informative visual representations, which are then synthesized with the outputs of LLMs to generate pertinent textual descriptions. A diverse array of datasets including COCO, Visual Genome, CC3M, CC12M, SBU, and LAION400M underpins the comprehensive pre-training regime of BLIP-2. These datasets provide a rich variety of image-text pairs, essential for training the model across a broad spectrum of visual representations and language generation tasks. The model's architecture and training approaches are designed to address the prohibitive costs associated with vision-and-language pre-training, offering a more efficient pathway to developing multimodal understanding and generation capabilities.
</details> 

### **xGen-MM (BLIP-3): An Open-Source Framework for Building Powerful and Responsible Large Multimodal Models**

xGen-MM (BLIP-3) is a comprehensive framework developed by Salesforce for training a series of open-source large multimodal models (LMMs) designed to excel in a variety of visual language tasks. It provides meticulously curated datasets, a streamlined training recipe, model architectures, and a suite of open LMMs capable of performing various visual language tasks. xGen-MM focuses on scalability, using a simplified architecture and a unified training objective to enable training on larger, more diverse datasets. The framework also includes a safety-tuned model to mitigate harmful behaviors and promote responsible AI development.

[![arXiv](https://img.shields.io/badge/arXiv-2408.08872-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.08872) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/Salesforce/xgen-mm-1-models-and-datasets-662971d6cecbf3a7f80ecc2e)  
Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, Shrikant Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, Ran Xu  

<p align="center">
<img src="https://github.com/user-attachments/assets/e6e166c8-871e-420c-bbf1-b64c3c22e06a" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

xGen-MM (BLIP-3), short for xGen-MultiModal, addresses limitations of previous open-source efforts by providing a complete ecosystem for LMM development. Central to its approach is the utilization of diverse, large-scale, and high-quality multimodal data, which enables xGen-MM to achieve competitive performance against both open-source and proprietary LMMs. Instead of relying on the intricate Q-Former architecture and multiple training objectives used in its predecessor, BLIP-2, xGen-MM streamlines the process by employing a more scalable vision token sampler (perceiver resampler) and unifying the training objective to a single auto-regressive loss on text tokens. This simplification enables larger-scale training and focuses the model on effectively learning from the rich multimodal context. Furthermore, xGen-MM incorporates safety measures, introducing a safety-tuned model with DPO to mitigate potential harmful behaviors like hallucinations and promote responsible AI development. By open-sourcing its models, datasets, and fine-tuning code, xGen-MM aims to empower the research community and foster advancements in the field of LMMs, making these powerful tools more accessible and encouraging further exploration of their capabilities.
</details>

### **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**

InstructBLIP enhances the BLIP-2 framework by introducing instruction tuning to its Query Transformer (Q-Former), enabling the model to extract instruction-aware visual features and achieve state-of-the-art zero-shot performance across diverse vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2305.06500v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.06500v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/InstructBLIP)  
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/5839e3a6-6fb8-469c-b84e-d60a851c1642" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**InstructBLIP**: represents an advanced step in the development of vision-language models through instruction tuning, building on the capabilities of the pre-trained BLIP-2 models. It integrates an image encoder, a large language model (LLM), and **a Query Transformer (Q-Former)**, which is specifically fine-tuned to bridge the visual and linguistic components while keeping the image encoder and LLM static. This architecture enables the extraction of instruction-aware visual features, enhancing the model's responsiveness to varied instructional contexts. Training InstructBLIP involves a careful selection of 26 datasets across 11 task categories, transformed into an instruction tuning format to foster the model's adaptability across a broad spectrum of vision-language tasks. The model employs a balanced sampling strategy and standard language modeling loss, augmented with OCR tokens for datasets involving scene texts, to fine-tune its instruction following capabilities. The unique approach of instruction-aware visual feature extraction through the Q-Former allows the model to tailor feature extraction to the specific requirements of the instruction, significantly improving performance across both seen and unseen tasks. Implementation details reveal the flexibility of InstructBLIP's architecture, which is easily adaptable to incorporate various LLMs, thanks to the modular design of the BLIP-2 framework. The model showcases state-of-the-art zero-shot performance across a wide range of vision-language tasks, outperforming previous models like BLIP-2 and Flamingo in zero-shot evaluations and achieving notable results when fine-tuned on specific downstream tasks. InstructBLIP's open-source availability and its performance across different benchmarks highlight its potential as a general-purpose vision-language model.
</details> 

### **KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models**

KOSMOS-1, a multimodal large language model, leverages a Transformer-based architecture enhanced with MAGNETO and XPOS to seamlessly process text and various modalities, aligning perception with language models through training on diverse web-scale multimodal corpora for enhanced zero-shot and few-shot learning capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2302.14045-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2302.14045) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/unilm)  
Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/33fd99a9-e89a-4905-8917-f03452fd5e6a" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**KOSMOS-1**: A transformative multimodal large language model, meticulously designed to harmonize the perception of general modalities with linguistic models, facilitating zero-shot learning, few-shot learning, and auto-regressive output generation. At its core, KOSMOS-1 employs a Transformer-based causal language model architecture, adept at processing both textual and various other modalities. This innovative approach is bolstered by key architectural components, including a Transformer-based decoder for input sequence handling, embedding modules for vector encoding of text and modalities, and the integration of **MAGNETO and XPOS** for architectural enhancements. These elements collectively enable the model to adeptly navigate and process multimodal information. The training regimen of KOSMOS-1 is distinguished by its comprehensive utilization of web-scale multimodal corpora, which encompasses monomodal data, cross-modal paired data, and interleaved multimodal data, emphasizing the next-token prediction tasks to optimize the log-likelihood of tokens. This methodology ensures a robust foundation for the model, enhancing its ability to understand and generate content across various modalities. Furthermore, the alignment techniques employed are particularly noteworthy; by leveraging interleaved image-text data, KOSMOS-1 aligns the perceptual capabilities of general modalities with language models in an unprecedented manner, thereby enriching the model's understanding and interpretative capacities. KOSMOS-1's training datasets, including The Pile, Common Crawl, English LAION-2B, LAION-400M, COYO-700M, and Conceptual Captions, are meticulously selected to serve dual purposes: fostering representation learning and language tasks through text corpora, and aligning perception with language models via image-caption pairs and interleaved data. This strategic selection of datasets not only bolsters the model's linguistic competencies but also significantly enhances its few-shot abilities, marking a significant milestone in the integration of perception and language models.
</details> 

### **KOSMOS-2: Grounding Multimodal Large Language Models to the World**

KOSMOS-2, extending the KOSMOS-1 architecture, incorporates grounded image-text pairs using discrete location tokens linked to text spans, effectively anchoring text to specific image regions, thereby enhancing multimodal understanding and reference accuracy.

[![arXiv](https://img.shields.io/badge/arXiv-2306.14824-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2306.14824) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/unilm/tree/master/kosmos-2) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ydshieh/Kosmos-2)  
Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/17420c9c-759d-4690-bfc8-e8d7792111e7" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**KOSMOS-2**: Built upon the foundational architecture of KOSMOS-1, it retains the Transformer-based causal language model architecture and training objectives, while introducing a significant innovation by incorporating grounded image-text pairs into its training regimen. This addition seeks to bridge the gap between visual and textual information, enabling a more cohesive understanding of multimodal content. The model differentiates itself by training on a web-scale dataset of grounded image-text pairs, known as GRIT, which includes continuous coordinates of bounding boxes translated into discrete location tokens. These tokens are intricately linked with text spans, creating a unified input representation that seamlessly integrates visual and textual elements. The training of KOSMOS-2 is extensive and multifaceted, utilizing grounded image-text pairs, monomodal text corpora, image-caption pairs, and interleaved image-text data to foster a robust learning environment. The model's training leverages a large batch size and employs the AdamW optimizer, running on 256 V100 GPUs. This process is augmented by instruction tuning with both vision-language and language-only instruction datasets, aiming to refine the model's understanding and processing capabilities across different modalities. The grounding technique is a pivotal aspect of KOSMOS-2, where **continuous coordinates of bounding boxes** are converted into **discrete location tokens**. These tokens are then linked with corresponding text spans, anchoring the textual output to specific visual inputs, enhancing the model's ability to refer to and describe particular image regions or objects with precision. KOSMOS-2's alignment techniques and fusion methods play a critical role in its ability to understand and refer to specific parts of an image directly, employing a unified input representation that combines image embeddings with grounded text and location tokens. This approach not only improves the model's referential accuracy but also its overall multimodal comprehension. The model is trained using a variety of datasets, including the specially created GRIT dataset for grounding capabilities, along with monomodal text corpora, image-caption pairs, and interleaved image-text data to bolster its language understanding, multimodal perception, and in-context learning abilities. Through these innovations, KOSMOS-2 represents a significant advancement in grounding multimodal large language models, offering enhanced capabilities in linking textual and visual information cohesively.
</details> 

### **ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models**

ConvLLaVA addresses the limitations of Vision Transformers (ViTs) in high-resolution Large Multimodal Models (LMMs) by replacing them with a hierarchical backbone, ConvNeXt, as the visual encoder. This architectural shift aims to reduce the computational burden caused by excessive visual tokens and quadratic complexity often associated with ViTs, especially when dealing with high-resolution images.

[![arXiv](https://img.shields.io/badge/arXiv-2405.15738-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.15738) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alibaba/conv-llava) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2405.15738)  
Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, Bo Zheng  

<p align="center">
<img src="https://github.com/user-attachments/assets/ad7e129a-f958-4b30-8327-7df509994bea" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

ConvLLaVA leverages the inherent information compression capabilities of ConvNeXt, a hierarchical convolutional neural network. ConvLLaVA, unlike traditional LMMs that rely on ViTs, employs a **five-stage ConvNeXt architecture** as its visual encoder. This encoder progressively compresses visual information across its stages, significantly reducing the number of visual tokens generated compared to ViT. The architecture mirrors other popular general LMMs like LLaVA, Qwen-VL, and VILA, consisting of a vision encoder (ConvNeXt), a large language model (LLM - Vicuna in this case), and a vision-language projector (a two-layer MLP). The ConvNeXt encoder processes the input image and generates latent visual embeddings. These embeddings are then projected into the embedding space of the LLM by the vision-language projector. Finally, the projected visual embeddings are concatenated with the text embeddings generated by the LLM's tokenizer, and this combined input is fed into the LLM. The entire model is trained using a language modeling loss. To further enhance ConvLLaVA's performance, the authors introduce two key optimizations: firstly, they update the pretrained ConvNeXt weights instead of freezing them, allowing the model to adapt to high-resolution inputs and improve the quality of visual representations. Secondly, they introduce an additional ConvNeXt stage, effectively creating a five-stage architecture (ConvNeXt†) that further compresses visual information, enabling the model to handle even higher resolutions (up to 1536x1536) while generating a manageable number of visual tokens (576). This hierarchical compression approach, combined with the linear spatial complexity of ConvNeXt, significantly reduces the computational burden on the LLM compared to ViT-based models, making ConvLLaVA a more efficient and scalable solution for high-resolution multimodal tasks.
</details>

### **Parrot: Multilingual Visual Instruction Tuning**

Parrot tackles the issue of "multilingual erosion" in Multimodal Large Language Models (MLLMs), where models trained primarily on English-centric data struggle to understand and respond in other languages. It achieves this by using textual guidance to align visual tokens with language-specific embeddings, effectively enhancing the model's multilingual capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2406.02539-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.02539) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/AIDC-AI/Parrot)  
Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, Han-Jia Ye  

<p align="center">
<img src="https://github.com/user-attachments/assets/467964a0-4ccc-4cec-802a-c93b310d3118" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Parrot builds upon the LLaVA framework, utilizing a pre-trained CLIP ViT-L/14 as the vision encoder and Qwen1.5-Chat as the LLM. The architecture consists of three main components: a vision encoder, a large language model (LLM), and a multilingual **Mixture-of-Experts (MoE)** module. The vision encoder processes the input image and generates visual features, which are then projected into the embedding space of the LLM using a learned projector. To address the multilingual challenge, Parrot introduces a novel textual guidance mechanism. It first calculates cross-attention between the class token of the visual features and the text embeddings derived from the input prompt. This cross-attention output is then fed into the MoE module's router, which predicts the probability of activating each language expert. Each expert is a specialized MLP trained to transform the English-biased visual embeddings into language-specific representations. The router selects the most relevant experts based on the input language, and their outputs are combined to generate the final language-specific visual embeddings. These embeddings are then combined with the original visual embeddings using a weighted sum, ensuring that the model retains its ability to process visual information effectively across different languages. This entire process allows Parrot to align visual tokens with textual embeddings at the language level, effectively mitigating multilingual erosion and enhancing the model's ability to understand and respond in multiple languages.
</details>

### **OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**

OMG-LLaVA presents a novel framework that unifies image-level, object-level, and pixel-level reasoning and understanding within a single Multimodal Large Language Model (MLLM). It leverages the power of a frozen universal segmentation model (OMG-Seg) for visual encoding and a Large Language Model (LLM) for text understanding and response generation, enabling a wide range of multimodal tasks within a single, elegant architecture.

[![arXiv](https://img.shields.io/badge/arXiv-2406.19389-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.19389) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/lxtGH/OMG-Seg) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2406.19389)  
Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, Shuicheng Yan  

<p align="center">
<img src="https://github.com/user-attachments/assets/c2830cc5-ab00-4c48-898e-a077cdc7b947" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

OMG-LLaVA consists of two main components: a frozen universal perception module (based on OMG-Seg) and a Large Language Model (LLM). The universal perception module is responsible for encoding the input image and visual prompts into three types of visual tokens: pixel-centric, object-centric, and object-centric derived from visual prompts. The pixel-centric tokens are generated by a **ConvNeXt-L based CLIP image encoder**, capturing dense image features. The object-centric tokens are generated by the OMG decoder, which takes learnable object queries and visual prompt queries as input and attends to the image features to extract object-level information. This decoder can handle point, box, and mask prompts by applying constraints on the attention masks. To bridge the gap between the frozen perception module and the LLM, a novel "perception prior embedding" strategy is introduced. This strategy fuses the image features with the object queries from the OMG decoder using a mask score derived from the segmentation masks and confidence scores. The resulting weighted object queries are then added to the image features to generate the pixel-centric visual tokens, providing the LLM with rich object-level information. The object-centric visual tokens are directly taken from the foreground object queries of the OMG decoder. Both types of visual tokens, along with the text instruction tokens, are fed into the LLM, which is responsible for understanding the user's intent and generating the appropriate response. The LLM outputs text responses and object-centric visual tokens, which are then decoded by the frozen OMG decoder to produce segmentation masks. This unified architecture allows OMG-LLaVA to perform a wide range of tasks, including image captioning, visual question answering, referring segmentation, reasoning segmentation, grounded conversation generation, and region captioning, all within a single model.
</details>

### **EVLM: An Efficient Vision-Language Model for Visual Understanding**

EVLM is an efficient multimodal language model designed to minimize computational costs while maximizing the model's ability to perceive visual signals comprehensively. It addresses the challenges of handling long sequences of visual signals, particularly in video data, by employing a cross-attention mechanism and hierarchical ViT features, achieving competitive performance in tasks like image and video captioning.

[![arXiv](https://img.shields.io/badge/arXiv-2407.14177-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.14177) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.14177)  
Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, Changyi Liu, Dewen Fan, Huihui Xiao, Jiahong Wu, Fan Yang, Size Li, Di Zhang  

<p align="center">
<img src="https://github.com/user-attachments/assets/87563a37-e65e-44d4-a0e1-aea452ae313c" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

EVLM is built upon the Flamingo architecture, incorporating a visual encoder, a large language model, and a Gated Cross-Attention Layer. To enhance visual perception, EVLM utilizes the 4.4B EVA2-CLIP-E-Plus model as the visual encoder, extracting hierarchical visual features by uniformly sampling 8 feature sequences from the last 40 layers of the transformer. These features are then sequentially fed into different Gated Cross-Attention layers of the Flamingo model. Unlike Flamingo, which uses a single media token image, EVLM replaces it with a set of 16 learnable tokens, aiming to capture visual features similar to Q-former. The attention mechanism is designed to allow each set of learnable tokens to interact only with the corresponding image, while text sequences interact only with the previous image in the multimodal sequence. This approach ensures efficient interaction between visual and textual information. For the language model, EVLM employs the Qwen-14B-Chat 1.0, chosen for its strong performance in content understanding and logical reasoning. A gated cross-attention layer is inserted before every transformer layer of the language model to condition it on visual inputs. To further enhance model effectiveness and scale trainable parameters, a Mixture of Experts (MoE) mechanism is applied to the Cross Attention layer. This involves replicating and segmenting the FFN of the base model into multiple fine-grained experts, with a routing layer selecting the appropriate set of experts for each token. The model undergoes a three-stage training process: multi-modal pre-training, multi-task continual pre-training, and multi-modal instruction fine-tuning. Pre-training focuses on cross-modal alignment and modeling intrinsic relationships within multimodal data, using a large-scale dataset of bilingual image-text captions and web-type multimodal data. Continual pre-training further enhances the model's visual question-answering ability, while instruction fine-tuning activates its instruction-following capabilities using a diverse range of high-quality instruction tuning data.
</details>

### **SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models**

SlowFast-LLaVA (SF-LLaVA) is a training-free video large language model that effectively captures both detailed spatial semantics and long-range temporal context in videos without requiring any additional fine-tuning on video data. It achieves this by leveraging a two-stream SlowFast design inspired by action recognition models, allowing it to process a larger number of frames and outperform existing training-free methods on various video benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2407.15841-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.15841) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.15841)  
Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, Afshin Dehghan  

<p align="center">
<img src="https://github.com/user-attachments/assets/6e1e2f43-86a7-42e3-998a-24bbd8f1c741" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

SF-LLaVA builds upon the LLaVA-NeXT framework and utilizes a two-stream approach, similar to SlowFast networks in action recognition, to process video inputs. The model first uniformly samples N frames from the input video. These frames are then processed independently by a visual encoder, such as CLIP-L, followed by a visual-language adapter for feature alignment. The resulting frame features are then fed into two separate pathways: Slow and Fast. **The Slow pathway** focuses on capturing detailed spatial semantics by processing a smaller number of frames (Nslow) at a higher spatial resolution (e.g., 8 frames with 24x24 tokens). It applies spatial pooling with a small stride (e.g., 1x2) to aggregate features and reduce the number of tokens. **The Fast pathway** focuses on capturing temporal context and motion cues by processing all N frames (Nfast = N) at a lower spatial resolution (e.g., 64 frames with 4x4 tokens). It applies aggressive spatial pooling to each frame to prioritize temporal information. The features from both pathways are then flattened and concatenated, forming a comprehensive video representation that balances spatial details and temporal context. This aggregated feature vector, along with the text prompt and question, is then fed into the LLM (LLaVA-NeXT) to generate the final answer. This training-free approach eliminates the need for expensive fine-tuning on video datasets, making SF-LLaVA highly efficient and adaptable to various video scenarios. The authors demonstrate the effectiveness of SF-LLaVA on three different video question-answering tasks (Open-Ended VideoQA, Multiple Choice VideoQA, and Text Generation) across eight benchmarks, showcasing its superior performance compared to existing training-free methods and even surpassing some state-of-the-art supervised fine-tuned video LLMs.
</details>

### **INF-LLaVA: High-Resolution Image Perception for Multimodal Large Language Models**

INF-LLaVA is a novel Multimodal Large Language Model (MLLM) designed to effectively process high-resolution images. It addresses the limitations of existing cropping-based and dual-encoder methods by introducing two innovative modules: Dual-perspective Cropping Module (DCM) and Dual-perspective Enhancement Module (DEM). DCM segments high-resolution images into sub-images from both local and global perspectives, preserving detailed and contextual information. DEM facilitates efficient interaction between local and global features, enhancing the model's understanding of complex visual relationships. Extensive evaluations demonstrate INF-LLaVA's superior performance on various benchmarks, establishing a new state-of-the-art in vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2407.16198-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.16198) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/WeihuangLin/INF-LLaVA) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.16198)  
Yiwei Ma, Zhibin Wang, Xiaoshuai Sun, Weihuang Lin, Qiang Zhou, Jiayi Ji, Rongrong Ji  

<p align="center">
<img src="https://github.com/user-attachments/assets/641027c4-a5eb-42e8-8486-b58f3508c553" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

INF-LLaVA pushes the boundaries of Multimodal Large Language Models (MLLMs) by tackling the critical challenge of high-resolution image perception. It aims to leverage the richness of detail present in high-resolution images without succumbing to the computational limitations imposed by traditional MLLM architectures. INF-LLaVA achieves this through a sophisticated approach that combines innovative cropping and feature enhancement techniques, resulting in a model capable of simultaneously capturing fine-grained local details and comprehensive global context. At the core of INF-LLaVA lies the Dual-perspective Cropping Module (DCM), a strategic cropping strategy that surpasses conventional approaches by integrating both local and global perspectives. This dual-perspective approach ensures that each extracted sub-image retains not only the intricate details essential for accurate analysis but also the broader contextual information crucial for understanding the relationships between objects. While local-perspective cropping preserves continuous visual information at high resolution, capturing the essence of individual objects and regions, global-perspective cropping leverages a unique interleaving technique to preserve the overall spatial relationships between objects within the high-resolution image. This balanced combination ensures that the model can perceive both the "trees" and the "forest," enabling a holistic understanding of the visual scene. To further enhance the model's understanding, INF-LLaVA introduces the Dual-perspective Enhancement Module (DEM). This module facilitates efficient and effective interaction between the local and global features extracted by the vision encoder, enriching the representation with multi-scale information. Instead of relying on computationally expensive cross-attention directly on high-resolution features, DEM employs a more resource-efficient strategy. It leverages 2D positional priors to concatenate global-perspective sub-image features back into the original image's shape, effectively recreating a high-resolution representation of the global context. These recombined features are then re-cropped from a local perspective, and cross-attention is performed between corresponding local and global sub-images to enhance global features with fine-grained local details. A symmetrical process enhances local features with global context. This meticulously designed interaction between local and global features ensures that the resulting representation is not only rich in detail but also cognizant of the broader context. The dual-enhanced features are then projected into a format compatible with the LLM through a linear connector. The LLM then processes the combined visual and textual information to generate a coherent and contextually relevant response. Through extensive evaluations on a diverse set of benchmarks, including ScienceQA-img, OKVQA, SEEDBench, MMBench, AI2D, LLaVA-Bench-in-the-wild, and MMMU, INF-LLaVA demonstrates its superior performance over existing MLLMs. Its ability to effectively handle high-resolution images while maintaining computational efficiency establishes a new state-of-the-art in the field. The open-source release of INF-LLaVA, along with its pretrained model and code, paves the way for further research and exploration of high-resolution image perception in multimodal large language models, pushing the boundaries of multimodal understanding and enabling the development of more powerful and versatile AI systems.
</details>


### **VILA²: VILA Augmented VILA**

VILA² (VILA-augmented-VILA) introduces a novel approach to address the limitations of data quantity and quality in training Visual Language Models (VLMs). Instead of relying on costly human annotation or distillation from proprietary models, VILA² leverages the VLM itself to iteratively refine and augment its pretraining data, leading to significant performance improvements and achieving state-of-the-art results on the MMMU leaderboard among open-sourced models.

[![arXiv](https://img.shields.io/badge/arXiv-2407.17453-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.17453) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.17453)  
Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jang Hyun Cho, Marco Pavone, Song Han, Hongxu Yin  

<p align="center">
<img src="https://github.com/user-attachments/assets/b7602734-1163-49aa-bf78-27ae42a520bd" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

VILA² employs a two-step iterative process: self-augmenting and specialist-augmenting. The self-augmenting loop focuses on enhancing the general knowledge of the VLM by using the model itself to re-caption its pretraining data. This process starts with an initial VLM (VILA0) trained on a dataset with typically short and brief captions, like COYO. VILA0 is then used to generate longer and more detailed captions for the same images, creating a synthetic dataset. This augmented dataset, combined with the original data, is used to train the next iteration of the VLM (VILA1). This loop can be repeated multiple times, with each iteration improving the caption quality and subsequently the VLM's performance. However, this self-augmentation process eventually reaches saturation. To overcome this limitation, VILA² introduces the **specialist-augmenting loo**p. This involves fine-tuning the self-augmented VLM on specific downstream tasks, creating specialist VLMs with expertise in areas like spatial awareness, OCR, and grounding. These specialists are then used to re-caption the pretraining data, focusing on their specific domain knowledge. The self-augmented VLM is then retrained on this specialist-recaptioned data, further boosting its performance. This approach leverages the synergy between the vast amount of data in pretraining and the specialized knowledge acquired during fine-tuning. The architecture of VILA² follows the standard auto-regressive VLM design, consisting of a large language model (LLM), a visual encoder, and an image-text projector. The authors experiment with different LLMs (Llama2-7B, Llama3-8B-Instruct, and Yi-34B) and visual encoders (SigLIP and InternViT-6B). They also introduce a 4x downsampling of visual tokens to reduce computational cost. The training process follows the typical three-stage paradigm: projector initialization, vision-language pre-training, and visual instruction-tuning. VILA² demonstrates significant performance improvements over previous state-of-the-art methods on various benchmarks, including general VQA, text-oriented VQA, general multimodal benchmarks, and image captioning. This highlights the effectiveness of the proposed self- and specialist-augmentation techniques in enhancing VLM training and achieving state-of-the-art results.
</details>

### **MiniCPM-V: A GPT-4V Level MLLM on Your Phone**

MiniCPM-V is a series of efficient Multimodal Large Language Models (MLLMs) designed for deployment on end-side devices like mobile phones and personal computers. The latest iteration, MiniCPM-Llama3-V 2.5, achieves performance comparable to GPT-4V, Gemini Pro, and Claude 3 while being significantly smaller and more efficient, demonstrating the feasibility of deploying powerful MLLMs on resource-constrained devices.

[![arXiv](https://img.shields.io/badge/arXiv-2408.01800-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.01800) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenBMB/MiniCPM-V) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/openbmb/MiniCPM-V-2_6)  
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, Maosong Sun  

<p align="center">
<img src="https://github.com/user-attachments/assets/d943871a-ca05-46d6-9572-7fe02dda1495" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

MiniCPM-V focuses on achieving a balance between performance and efficiency, crucial for real-world applications on end-side devices. The model architecture consists of three key modules: a visual encoder, a compression layer, and an LLM. For the visual encoder, MiniCPM-V utilizes SigLIP SoViT-400m/14, chosen for its efficiency and effectiveness. To handle high-resolution images with varying aspect ratios, the model employs an adaptive visual encoding approach. This involves dividing the input image into slices that better match the ViT's pre-training settings in terms of resolution and aspect ratio. A score function is used to select the optimal partition of slices, ensuring a good match with the ViT's pre-training. Each slice is then resized proportionally and interpolated to fit the ViT's input size. After visual encoding, each slice is represented by 1024 tokens, resulting in a large number of tokens for multiple slices. To address this, a token compression module is employed, using one-layer cross-attention with a moderate number of queries to compress the visual tokens of each slice into 64 or 96 tokens. This significantly reduces the computational cost and memory footprint, making the model suitable for end-side deployment. A spatial schema is also introduced to indicate the position of each slice relative to the whole image, further enhancing the model's understanding of spatial relationships. The compressed visual tokens, along with the text input, are then fed into the LLM, which is based on MiniCPM 2B for earlier versions and Llama3-Instruct 8B for MiniCPM-Llama3-V 2.5. The training process consists of three phases: pre-training, supervised fine-tuning, and RLAIF-V (Reinforcement Learning from AI Feedback for Vision). Pre-training aims to align the visual modules with the LLM's input space and learn foundational multimodal knowledge. It involves three stages: warming up the compression layer, extending the input resolution of the visual encoder, and training the visual modules with the adaptive visual encoding strategy. Supervised fine-tuning further enhances the model's knowledge and interaction capabilities using high-quality visual question answering datasets. The SFT data is categorized into two parts: one focusing on basic recognition capabilities and the other on generating detailed responses and following instructions. Finally, RLAIF-V is employed to mitigate the hallucination problem common in MLLMs. This involves generating multiple responses for an instruction, evaluating their correctness using a divide-and-conquer strategy, and then optimizing the model using Direct Preference Optimization (DPO) on a preference dataset. MiniCPM-V demonstrates impressive performance on various benchmarks, including general multimodal benchmarks, OCR benchmarks, and multilingual multimodal interaction, while being efficient enough for deployment on mobile phones. This highlights the potential of pushing the boundaries of end-side MLLMs and bringing powerful AI capabilities to user devices.
</details>

### **LLaVA-OneVision: Easy Visual Task Transfer**

LLaVA-OneVision is a family of open large multimodal models (LMMs) designed to excel in various computer vision scenarios, including single-image, multi-image, and video understanding. It pushes the performance boundaries of open LMMs by consolidating insights from the LLaVA-NeXT blog series, focusing on data, models, and visual representations. Notably, LLaVA-OneVision demonstrates strong transfer learning capabilities, enabling it to excel in video understanding tasks by leveraging knowledge learned from image data.

[![arXiv](https://img.shields.io/badge/arXiv-2408.03326-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.03326) [![Website](https://img.shields.io/badge/🌐-Website-blue)](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2408.03326)  
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li  

<p align="center">
<img src="https://github.com/user-attachments/assets/abe36db3-571d-4068-b532-7512d4a5fcc5" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

LLaVA-OneVision inherits the minimalist design of the LLaVA series, aiming to effectively leverage pre-trained capabilities of both the LLM and the visual model while facilitating strong scaling. The architecture consists of three key components: a large language model (LLM), a vision encoder, and a projector. The authors choose Qwen-2 as the LLM due to its strong language capabilities and various model sizes available. For the vision encoder, they opt for SigLIP, which has shown to yield higher LMM performance among open vision encoders. A 2-layer MLP is used as the projector to map image features into the word embedding space, creating a sequence of visual tokens. The model utilizes a flexible visual representation strategy called Higher AnyRes, which builds upon the original AnyRes strategy introduced in LLaVA-NeXT. This strategy involves dividing the input image into crops, each with a resolution suitable for the vision encoder, and then applying bilinear interpolation to reduce the number of tokens per crop if needed. This allows the model to handle high-resolution images and videos efficiently while preserving important visual details. The specific configuration of **Higher AnyRes** is adapted for different scenarios: single-image, multi-image, and video. For single-image data, a large maximum spatial configuration is used to maintain the original image resolution and a large number of visual tokens are allocated to effectively represent the visual signal. For multi-image data, only the base image resolution is considered, eliminating the need for multi-crop and saving computational resources. For video data, each frame is resized to the base image resolution and bilinear interpolation is used to reduce the number of tokens per frame, allowing for the processing of a larger number of frames. The training process follows a three-stage curriculum learning approach: language-image alignment, high-quality knowledge learning, and visual instruction tuning. The first stage focuses on aligning visual features with the LLM's embedding space using the LLaVA align dataset. The second stage refines and enhances the model's knowledge base using high-quality data from three major categories: re-captioned detailed description data, document/OCR data, and Chinese and language data. The final stage involves visual instruction tuning, where the model is trained on a diverse set of visual tasks with preferred responses. This stage is further divided into two phases: single-image training and OneVision training. Single-image training focuses on single-image scenarios, while OneVision training expands the model's capabilities to multi-image and video scenarios, enabling task transfer and emerging capabilities. LLaVA-OneVision demonstrates state-of-the-art performance on various benchmarks, including single-image, multi-image, and video tasks, showcasing its effectiveness and versatility in handling diverse visual scenarios.
</details>

### **VITA: Towards Open-Source Interactive Omni Multimodal LLM**

VITA is the first open-source Multimodal Large Language Model (MLLM) capable of simultaneously processing and analyzing video, image, text, and audio modalities while offering an advanced multimodal interactive experience. It addresses the limitations of existing open-source models, which often excel in either understanding or interaction but rarely both, by integrating architectural innovations with advanced training and development strategies.

[![arXiv](https://img.shields.io/badge/arXiv-2408.05211-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.05211) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/VITA-MLLM/VITA) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/VITA-MLLM)  
Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, Xing Sun

<p align="center">
<img src="https://github.com/user-attachments/assets/94e2b781-0c86-47df-ac18-76ebc71bb349" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

VITA starts with the Mixtral 8x7B model as its language foundation, chosen for its strong performance and sparse mixture of experts (SMoE) architecture. To enhance its Chinese language capabilities, the vocabulary is expanded with Chinese terms, and the model undergoes bilingual instruction tuning using a high-quality bilingual text corpus. This ensures proficiency in both Chinese and English. For visual modality, VITA employs InternViT-300M-448px as the visual encoder, processing images at 448x448 resolution and generating 256 tokens after passing through a two-layer MLP visual connector. High-resolution images are handled using a dynamic patching strategy, while videos are treated as special cases of images, with frame sampling based on video length. For audio modality, a Mel Filter Bank block is used to process the input audio, followed by 4xCNN downsampling layers and a 24-layer transformer, resulting in 25 tokens for every 2 seconds of audio. A two-layer MLP serves as the audio-text modality connector. The training pipeline consists of three stages: LLM instruction tuning, multimodal alignment, and multimodal instruction tuning. LLM instruction tuning focuses on enhancing the base LLM's bilingual capabilities. Multimodal alignment aims to bridge the representation gap between text and other modalities by training individual encoders and connectors for each modality. This involves collecting and curating a large-scale, high-quality multimodal dataset, including image descriptions, general image QA, OCR and diagram data, general video descriptions, general video QA, and pure text data. Multimodal instruction tuning further refines the model's ability to follow instructions and understand different modalities. A specially designed state token is introduced to distinguish the type of input query (effective audio, noisy audio, or text), enabling non-awakening interaction during inference. To achieve natural multimodal human-computer interaction, VITA introduces two key innovations: non-awakening interaction and audio interrupt interaction. These are implemented using a duplex pipeline during deployment. Two VITA models run concurrently: one for generating responses to user queries (Generation model) and the other for monitoring environmental audio (Monitoring model). The Monitoring model uses SileroVAD for voice activity detection and filters out noisy audio based on the state token. If an effective audio query is detected, the Monitoring model interrupts the Generation model, consolidates the historical context, and responds to the latest query. The two models then swap identities, ensuring continuous monitoring and seamless interaction.VITA demonstrates strong performance on various unimodal and multimodal benchmarks, showcasing its robust foundational capabilities in multilingual, vision, and audio understanding. While still lagging behind closed-source counterparts in certain areas, VITA represents a significant step towards open-source interactive omni-modal LLMs, paving the way for future research and development in this field.
</details>

### **EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders**

EAGLE is a family of open-source Multimodal Large Language Models (MLLMs) that leverage a mixture of vision encoders to achieve state-of-the-art performance on various benchmarks, particularly in tasks involving OCR and document understanding. The study focuses on systematically exploring the design space of MLLMs with multiple vision encoders, aiming to identify optimal design choices and improve MLLM perception.

[![arXiv](https://img.shields.io/badge/arXiv-2408.15998-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.15998) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/NVlabs/EAGLE) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/NVEagle/Eagle-X5-13B-Chat)  
Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, Guilin Liu  

<p align="center">
<img src="https://github.com/user-attachments/assets/4e057a78-3fad-4a04-9a05-0f5361a8255b" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

EAGLE builds upon the LLaVA architecture, consisting of a large language model, a vision encoder, and a projection layer. The core innovation lies in integrating multiple vision experts, each pre-trained on different tasks and resolutions, to enhance the model's ability to perceive and comprehend diverse visual information. The study explores various aspects of this design space, including high-resolution adaptation, fusion paradigms, and optimal encoder combinations. It introduces a Pre-Alignment training stage to address representational inconsistencies between vision-focused encoders and language tokens. The training process consists of three progressive stages: vision-language pre-alignment, joint-projector training, and supervised fine-tuning. EAGLE achieves state-of-the-art performance on various benchmarks, demonstrating significant advantages in OCR and document understanding tasks. The study highlights the importance of systematic design space exploration and the effectiveness of combining multiple vision experts with a streamlined fusion strategy and a pre-alignment training stage for building high-performing MLLMs.
</details>

### **Florence-2: A Deep Dive into its Unified Architecture and Multi-Task Capabilities**

Florence-2 presents a significant advancement in vision foundation models, aiming to achieve a single, versatile representation capable of handling a wide spectrum of vision and vision-language tasks through a unified, prompt-based approach. Unlike previous models that often specialize in specific tasks, Florence-2 is designed to be a generalist, adept at performing tasks with simple text instructions, similar to how Large Language Models (LLMs) operate.

[![arXiv](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2311.06242) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/gokaygokay/Florence-2)  
Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan  

<p align="center">
<img src="https://github.com/user-attachments/assets/f9c1f95b-ba6a-4a55-bf52-fa043b339d27" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Florence-2 lies a sophisticated architecture comprised of two key components: an image encoder and a multi-modality encoder-decoder. The image encoder, powered by the powerful DaViT architecture, transforms the input image into a sequence of visual token embeddings, effectively capturing the visual information. These visual embeddings are then combined with text embeddings derived from task-specific prompts. This fusion of visual and linguistic information is processed by a standard transformer-based multi-modality encoder-decoder. This component acts as the brain of the model, meticulously analyzing the combined input and generating the desired output in textual form. This unified architecture, with a single set of parameters governing various tasks, eliminates the need for task-specific modifications, leading to a streamlined and efficient model. This design philosophy mirrors the trend in the NLP community, where models with consistent underlying structures are preferred for their versatility and ease of development. Florence-2's capabilities span a multitude of tasks, showcasing its remarkable adaptability. It excels at generating detailed image captions, capturing the essence of an image through rich textual descriptions. Its prowess extends to visual grounding, accurately pinpointing specific objects or regions within an image based on textual phrases. Florence-2 also demonstrates impressive performance in open-vocabulary object detection, identifying objects by their names, even if those objects were not part of its training data. This capability highlights the model's ability to generalize its knowledge and understand novel visual concepts. Furthermore, Florence-2 tackles dense region captioning, providing detailed descriptions for multiple regions within an image, and even performs optical character recognition (OCR), extracting text from images. This broad range of capabilities makes Florence-2 a powerful tool for numerous applications, pushing the boundaries of multimodal understanding in AI.
</details>

### **MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning**

MULTIINSTRUCT leverages the OFA model as its foundation, employing a Transformer-based sequence-to-sequence architecture and instruction tuning techniques on a diverse dataset, effectively aligning text and image tokens within a unified space for enhanced multi-modal zero-shot learning.

[![arXiv](https://img.shields.io/badge/arXiv-2212.10773-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2212.10773) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vt-nlp/multiinstruct)  
Zhiyang Xu, Ying Shen, Lifu Huang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/bedfc8b1-7aff-44af-b605-4470ad030bdf" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MULTIINSTRUCT**: introduces a novel approach to enhance multi-modal zero-shot learning by leveraging instruction tuning, built upon the foundation of the **OFA (Omnipotent Fast Adapters)** as its core pre-trained multi-modal model. This model adopts a Transformer-based sequence-to-sequence architecture that efficiently encodes a mix of instructions, text, images, and bounding boxes within a unified token space. Such a design enables MULTIINSTRUCT to process and interpret a wide range of input types, including optional images, through a comprehensive encoder-decoder framework. The encoder component is dedicated to processing the diverse inputs and instructions, while the decoder is tasked with generating the corresponding outputs. At the heart of MULTIINSTRUCT's training methodology is the innovative use of the model-specific MULTIINSTRUCT dataset, alongside instruction tuning techniques that incorporate instances from multiple tasks. This approach involves a combination of random shuffling and sampling of instruction templates for batch training, significantly enriching the learning process. Furthermore, the model explores advanced transfer learning strategies through Mixed Instruction Tuning and Sequential Instruction Tuning, utilizing the NATURAL INSTRUCTIONS dataset. This strategy not only enhances the model's adaptability across a wide spectrum of multi-modal tasks but also boosts its performance in zero-shot learning scenarios. The alignment techniques employed by MULTIINSTRUCT, such as byte-pair encoding and VQ-GAN, play a crucial role in aligning text and image tokens within a unified vocabulary. This seamless integration allows the model to effectively process and interpret various types of inputs and outputs. The use of a unified sequence-to-sequence architecture facilitates a deeper integration and alignment of vision and language modalities, underscoring the model's innovative approach to bridging the gap between different types of data. The datasets used for training and fine-tuning, namely MULTIINSTRUCT and NATURAL INSTRUCTIONS, are specifically chosen to bolster the model's capabilities in handling multi-modal tasks and instructions, showcasing its versatility and effectiveness in enhancing multi-modal zero-shot learning.
</details> 

### **MouSi: Poly-Visual-Expert Vision-Language Models**

MouSi pushes the boundaries of VLMs by incorporating multiple visual experts like CLIP and SAM, utilizing a poly-expert fusion network to combine their outputs and interface with powerful LLMs like Vicuna, thereby enabling a more comprehensive understanding and processing of visual information.

[![arXiv](https://img.shields.io/badge/arXiv-2401.17221-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.17221) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/fudannlplab/mousi)  
Xiaoran Fan, Tao Ji, Changhao Jiang, Shuo Li, Senjie Jin, Sirui Song, Junke Wang, Boyang Hong, Lu Chen, Guodong Zheng, Ming Zhang, Caishuang Huang, Rui Zheng, Zhiheng Xi, Yuhao Zhou, Shihan Dou, Junjie Ye, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, Yu-Gang Jiang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/7e09c9d8-4c18-4970-9a24-b5e538285a72" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MouSi**: Represents an innovative approach to Vision-Language Models (VLMs) by integrating multiple visual experts into a unified architecture, aiming to surpass the limitations inherent to models reliant on a singular visual component. This architecture leverages a poly-expert fusion network, which incorporates outputs from varied visual experts, such as CLIP for image-text matching and SAM for image segmentation. This network facilitates an efficient interface with pre-trained Large Language Models (LLMs), notably utilizing a model like Vicuna v1.5. MouSi distinguishes itself by employing a multi-expert visual encoder that selects relevant experts from a pool, and it features two types of **poly-expert fusion networks: a projection fusion method and a Q-Former fusion method.** The training methodology of MouSi is characterized by a two-phase approach. Initially, during the pre-training phase, both the text-only LLM and the multi-expert encoder are kept static, with the training focus squarely on the poly-visual fusion network. Subsequently, in the fine-tuning phase, the LLM is activated for training in conjunction with the poly-visual fusion network, using high-quality supervised datasets. This methodology ensures that MouSi benefits from robust pre-existing language models while simultaneously enhancing its capability to process and integrate complex visual information. For alignment and fusion of the multimodal inputs, MouSi employs its poly-expert fusion network to amalgamate the outputs from the various visual experts, aligning them with the vision input tokens. This alignment is critical for encoding vision and text cohesively, a process facilitated by either the projection fusion method or the more complex Q-Former fusion method. These methods allow for the effective compression of multi-channel visual information into a format that can be efficiently processed alongside textual data. The datasets used in MouSi's training regimen include LCS-558K and the LAION-CC-SBU collection for pre-training, aimed at aligning text and image representation spaces, and diverse, high-quality SFT datasets for fine-tuning, enhancing the model's performance across a broad spectrum of multimodal tasks.
</details> 

### **LaVIN: Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models**

LaVIN offers an efficient and cost-effective approach to vision-language instruction tuning by employing a Mixture-of-Modality Adapter (MM-Adapter), significantly reducing trainable parameters and enabling a streamlined optimization process for LLMs without extensive pre-training.

[![arXiv](https://img.shields.io/badge/arXiv-2305.15023v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.15023v3) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/luogen1996/lavin)  
Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, Rongrong Ji
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/8afc8259-fa72-4e52-8080-a4ea12208e32" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LaVIN**: This model introduces the Mixture-of-Modality Adaptation (MMA) learning regime, a pioneering method that leverages **lightweight adapters** to fine-tune LLMs for vision-language (VL) instruction tasks. The core of LaVIN's architecture is the **Mixture-of-Modality Adapter (MM-Adapter)**, which connects the image encoder to the LLM using minimal adaptation modules, allowing for a streamlined optimization of the multimodal LLM through a relatively small number of parameters. The training methodology of LaVIN is notably efficient, employing the MMA strategy to fine-tune only the inserted adapters, thus significantly reducing the optimized parameter count to between three to five million. This method substantially lowers both training time and storage requirements, circumventing the need for additional VL pre-training. The MM-Adapter is instrumental in facilitating the seamless transition between single- and multi-modal instructions, thereby enhancing the model's adaptability to various VL tasks. Additionally, it employs a dynamic routing function that adjusts adaptations for input features, enabling an effective integration of vision and text embeddings. LaVIN's performance and versatility are further demonstrated through its application on diverse datasets, including ScienceQA, Alphaca-52k, and LLaVA-158k. ScienceQA is utilized to assess the model's multimodal question-answering capabilities, while the Alphaca-52k (text-only) and LLaVA-158k (text-image pairs) datasets are leveraged to refine and expand LaVIN's functionality as a multimodal chatbot. This strategic use of datasets underscores LaVIN's advanced vision-language understanding, illustrating its potential to significantly contribute to the field of multimodal learning and interaction.
</details> 

### **Nous-Hermes-2-Vision - Mistral 7B**

Nous-Hermes-2-Vision builds upon OpenHermes-2.5 by integrating the efficient SigLIP-400M vision encoder and incorporating a custom dataset with function calling capabilities, enabling it to not only understand visual and textual information but also extract specific text from images, advancing its functionality as a Vision-Language Action Model.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha)  
This project is led by qnguyen3 and teknium.
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Nous-Hermes-2-Vision**: Represents a notable advancement in the realm of Vision-Language Models, marking its distinction through the integration of two key enhancements that elevate its capabilities beyond traditional models. This model is an evolution from its predecessor, **OpenHermes-2.5-Mistral-7B**, and distinguishes itself by incorporating the **SigLIP-400M** for significantly improved performance and efficiency, moving away from the standard reliance on larger 3B vision encoders. Additionally, it introduces a custom dataset that includes function calling capabilities, transforming it into a more dynamic Vision-Language Action Model. The training of Nous-Hermes-2-Vision utilized a diverse dataset comprising 220K images from LVIS-INSTRUCT4V, 60K from ShareGPT4V, 150K private function calling data, and 50K conversations from teknium's OpenHermes-2.5. Such a varied dataset ensures the model's proficiency across a broad spectrum of vision-language tasks, including object recognition, instruction following, and conversational understanding. The model's innovative approach to integrating vision and language, particularly through the use of custom datasets for function calling, allows for encoding vision and text together in a way that supports action-oriented tasks and automation. A key feature of Nous-Hermes-2-Vision is its ability to interact with images to extract valuable text information from visual content, thus enabling detailed analyses and responses in natural language. This capability is underscored by the model's utilization of the SigLIP-400M, opting for a more lightweight and efficient architecture while enhancing performance in vision-language tasks. The model is further enriched with a custom dataset that includes **function calling**, allowing for the extraction of written information from images through specific tags, thus broadening its application scope for developers and researchers alike. Despite its innovative features, early usage of Nous-Hermes-2-Vision has revealed some challenges, such as hallucinations and spamming of EOS tokens. Recognizing these issues, the research team, led by Quan Nguyen and Teknium, has committed to releasing an updated version to address these problems, demonstrating their dedication to refining the model's capabilities.
</details> 

### **TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**

TinyGPT-V prioritizes efficiency in multimodal large language models by combining a compact EVA-ViT visual encoder with linear projection layers and the powerful Phi-2 language model, achieving robust performance in vision-language tasks despite its smaller size.

[![arXiv](https://img.shields.io/badge/arXiv-2312.16862v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.16862v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/DLYuanGod/TinyGPT-V) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/llizhx/TinyGPT-V)  
Zhengqing Yuan, Zhaoxu Li, Lichao Sun
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/3e7c93bc-7963-4c2e-b207-226a03d152ca" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**TinyGPT-V**: introduces a compact yet powerful architecture tailored for efficient multimodal large language model applications, leveraging small backbones for streamlined processing. This model integrates a visual encoder, specifically EVA of Vision Transformer (ViT), with **linear projection layers** and the Phi-2 language model, constituting its core components. The visual encoder remains inactive during training, focusing on image resolution adjustments across various stages to enhance image understanding. The **linear projection layers**, particularly with the incorporation of the **Q-Former layer** from BLIP-2, aim to efficiently embed visual features into the language model, reducing the number of parameters needing training. The Phi-2 large language model backbone, a 2.7 billion-parameter model, excels in reasoning and language comprehension, effectively handling vision-language operations including spatial location tasks through textual bounding box depictions. The training of TinyGPT-V unfolds across four stages: warm-up, pre-training, instruction fine-tuning, and multi-task learning. Each stage is meticulously designed to progressively enhance the model's capabilities in understanding and generating language based on visual inputs, with a special emphasis on human-like learning and conversation abilities in later stages. The use of datasets such as LAION, CC3M, SBU, and more, across these stages, supports the model's development in vision-language understanding, generation, and task execution like visual question answering and image captioning. A noteworthy aspect of TinyGPT-V's architecture is the implementation of normalization techniques and LoRA (Low-Rank Adaptation) to stabilize training and optimize the model's performance across different modalities. Addressing challenges like NaN or INF values in multimodal data computation, these mechanisms enhance training stability and efficiency. Furthermore, the model employs a multi-task instruction template to manage task ambiguity, utilizing MiniGPT-v2 tokens for task-specific instructions, facilitating precise and accurate task execution.
</details> 

### **CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding**

CoVLM distinguishes itself by using novel communication tokens to enable dynamic interaction between its CLIP ViT-L image encoder, YOLOX detection network, and Pythia language model, facilitating sophisticated communication for superior compositional reasoning in vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03354v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03354v1)  
Junyan Li, Delin Chen, Yining Hong, Zhenfang Chen, Peihao Chen, Yikang Shen, Chuang Gan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/80e807cb-c2cf-491a-a3b4-1223afde1981" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CoVLM**: This model is distinct in its approach, employing a novel set of **communication tokens** that facilitate dynamic interaction between a vision encoder, detection network, and a language model (LLM). The architecture of CoVLM integrates a CLIP ViT-L image encoder and a YOLOX detection network, alongside a pre-trained Pythia model for language processing. These components work in tandem to guide the LLM in composing visual entities and relationships within the textual context, enhancing the model's ability to dynamically communicate with the vision encoder and detection network. CoVLM is pre-trained on a diverse and extensive image-text dataset comprising 97 million image-text pairs, drawn from a variety of sources. This extensive dataset supports the model's grounding pipeline, which is crucial for associating text spans with their corresponding visual entities in images. The model utilizes special communication tokens for facilitating iterative communication between its vision and language components, enabling a sophisticated form of top-down and bottom-up communication. This communication is key to achieving high performance in vision-language tasks, as it allows the model to seamlessly integrate and interact between language tokens and visual embeddings. The datasets employed for pre-training, such as COCO, CC3M, CC12M, Visual Genome, SBU, and LAION400M, are meticulously selected to enhance the model's ability to ground image-text pairs effectively. This strategic choice is aimed at facilitating the association of textual descriptions with their corresponding visual entities, thereby improving the model's overall performance across a range of multimodal tasks. CoVLM's innovative approach to integrating visual detection networks with LLMs enables a new level of compositional reasoning, setting it apart from previous vision-language models.
</details> 

### **GLaMM: Pixel Grounding Large Multimodal Model**

GLaMM excels in pixel-level grounding by utilizing a five-component architecture encompassing global and regional image encoders, an LLM, a grounding image encoder, and a pixel decoder, allowing for comprehensive visual understanding and precise object localization within images.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03356-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03356) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mbzuai-oryx/groundingLMM)  
Hanoona Rasheed, Muhammad Maaz, Sahal Shaji Mullappilly, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Erix Xing, Ming-Hsuan Yang, Fahad S. Khan  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/ccb22206-6a48-4b77-8cc1-094fe86d72fd" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**GLaMM**: At its core, GLaMM comprises five essential components: the **Global Image Encoder, Region Encoder, Language Model (LLM), Grounding Image Encoder, and Pixel Decoder**. This architecture is designed to facilitate a wide range of interactions with visual content, from scene-level understanding through the Global Image Encoder, to detailed region-level interpretations via the Region Encoder, and down to precise pixel-level object grounding with the Grounding Image Encoder. The Pixel Decoder component further enriches the model's capabilities by generating **segmentation masks**, enabling GLaMM to respond to both textual and visual prompts with high fidelity. The training methodology of GLaMM involves a dual-pathway approach, encompassing both automated and manual data annotation pipelines to create the Grounding-anything Dataset (GranD). GranD is pivotal for the model's training, especially for its Grounded Conversation Generation (GCG) task, offering a rich set of 7.5 million unique concepts grounded in 810 million regions, complete with segmentation masks. This dataset not only supports the pretraining and fine-tuning phases of GLaMM but also underlines its unique ability to generate grounded conversations that are contextually relevant to the visual stimuli. Alignment techniques within GLaMM utilize a vision-to-language (V-L) projection layer, facilitating the mapping of image features into the language space, thereby ensuring effective text-image alignment. Furthermore, the model employs a language-to-prompt (L-P) projection layer, transforming text embeddings related to segmentation into the decoder space. This dual-projection system allows for an integrated encoding of vision and text, bolstering GLaMM's capacity for pixel-level grounding and positioning it as a significant advancement in the field of multimodal interactions.
</details> 

### **COSMO: COntrastive Streamlined MultimOdal Model with Interleaved Pre-Training**

COSMO presents a streamlined multimodal framework by combining a Vision Transformer with a partitioned Large Language Model, optimizing the processing of interleaved data sequences through a combination of language modeling and contrastive loss functions.

[![arXiv](https://img.shields.io/badge/arXiv-2401.00849v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.00849v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](http://fingerrec.github.io/cosmo)  
Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, Mike Zheng Shou
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0c256daa-1573-4110-a665-5927ee2e293f" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**COSMO**: This framework is distinctive for its architecture that merges a visual encoder, leveraging the Vision Transformer (ViT) from Open-CLIP, with a partitioned Large Language Model (LLM). The LLM is systematically divided into segments dedicated to unimodal text processing and multimodal data handling, aiming to streamline the overall processing of interleaved data sequences. The introduction of an additional contrastive loss component stands out as a strategy to improve performance across both classification and generation tasks. Training of COSMO is carried out through a unique combination of language modeling loss and contrastive loss, focusing on the efficient management of interleaved text and visual sequences. This process is optimized with the use of the AdamW optimizer, a cosine learning rate schedule, and the implementation of DeepSpeed fp16 precision, distributed across 128 NVIDIA V100 GPUs. The partitioning strategy of the LLM into dedicated components is a testament to the framework's commitment to computational efficiency and efficacy in handling extensive data sequences. The model's alignment techniques are notably advanced, featuring a learnable query that facilitates global attention across all tokens, alongside an additional query for **Text Fusion Layers**, optimizing the model's understanding of token sets and enhancing image-text alignment through contrastive loss. **The gated cross-attention layers** for multimodal fusion introduce a significant reduction in learnable parameters by introducing bottlenecks in input and output feature channels. This method of lightweight fusion is pivotal in integrating visual information for precise next-token prediction. COSMO's training leverages a diverse array of datasets including CC3M, SBU, LAION400M, DataComp1B, MMC4, WebVid, and Howto-Interlink7M. The introduction of Howto-Interlink7M, in particular, underscores the model's innovative approach to improving video-language understanding through high-quality annotated captions, demonstrating its effectiveness across 14 diverse downstream tasks.
</details> 

### **FireLLaVA**

FireLLaVA breaks new ground by combining the CodeLlama 34B Instruct model for advanced language understanding with a CLIP-ViT-based visual interpretation component, training on a unique dataset incorporating bounding box labels and captions to excel in visual language conversations.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/fireworks-ai/FireLLaVA-13b)   

<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FireLLaVA**: As the first of its kind within the LLaVA lineage, FireLLaVA integrates a dual-component architecture that leverages the CodeLlama 34B Instruct model for nuanced language understanding and a visual interpretation component akin to OpenAI's CLIP-ViT. This model is distinctive for its use of bounding box labels and captions to generate visual language conversations, a method that underscores its innovative approach to multi-modal training. The training regimen for FireLLaVA is meticulously crafted, utilizing 588K lines of visual question answering and conversation data. This dataset amalgamates permissive original LLaVA data with newly generated data from Fireworks.ai, demonstrating a unique approach to instruction fine-tuning that enhances the model's ability to comprehend and articulate responses that bridge textual and visual inputs. The integration of bounding box labels and captions not only serves as a mechanism for generating training data but also facilitates the alignment of text and image data, a crucial step in achieving coherent multi-modal understanding. Although the specific methods employed for alignment fusion within FireLLaVA's architecture remain under-described, it is inferred that embedding fusion plays a critical role in synthesizing vision and text inputs. By drawing on original LLaVA training materials and Fireworks.ai's proprietary data, FireLLaVA sets a precedent for the development of VLMs capable of navigating the complexities of commercial applications. This model embodies a significant advancement in the field of visual language modeling, offering insights into the potential of OSS models to contribute to the evolving landscape of multi-modal AI research and deployment.
</details> 

### **u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model**

u-LLaVA introduces a novel projector-based architecture that unifies multi-modal tasks by connecting specialized expert models with a central Large Language Model (LLM), enabling seamless modality alignment and efficient multi-task learning through a two-stage training approach.

[![arXiv](https://img.shields.io/badge/arXiv-2311.05348-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.05348) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OPPOMKLab/u-LLaVA)  
Jinjin Xu, Liwu Xu, Yuzhe Yang, Xiang Li, Yanchun Xie, Yi-Jie Huang, Yaqian Li
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/dcb6b046-fa56-4a02-9123-2ef2185c635a" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**u-LLaVA**: Represents a pioneering approach in the integration of Large Language Models (LLMs) with specialized expert models to address a wide array of multi-modal tasks. This architecture is designed to leverage the strengths of LLMs as a central hub, facilitating seamless modality alignment and multi-task learning. Through a novel **projector-based structure** that incorporates CLIP's Vision Transformer (ViT-L/14) and LLaMA2, u-LLaVA introduces a flexible framework capable of handling diverse modalities and tasks. The system integrates special tokens for modality and task expressions, alongside dedicated modules for segmentation, grounding, and in-painting, to enrich its multi-modal capabilities. The training methodology of u-LLaVA is executed in two distinct stages, beginning with a coarse-grained alignment to ensure the alignment of representation spaces across different modalities. This foundational step is crucial for establishing a common ground for further, more nuanced task-specific adaptations. Following this, a fine-grained alignment phase focuses on the refinement of task-specific instruction data, optimizing the model's performance for targeted applications. This dual-stage training approach ensures that u-LLaVA can efficiently adapt to a variety of tasks with minimal additional training requirements. Central to u-LLaVA's effectiveness is its innovative use of projector-based alignment techniques and fusion methods, which enable the integration of visual and textual representations within the LLM's framework. By mapping hidden states and text embeddings through projectors, u-LLaVA facilitates modality fusion, leveraging the extensive knowledge embedded within LLMs for complex task solving. The datasets utilized for training, including LLaVA CC3M, Conversation-58K, Detail-23K, and others, are meticulously curated to support the model's versatile capabilities across tasks such as image captioning, video captioning, visual question answering (VQA), referential expression comprehension (RES), semantic segmentation, and salient object detection/segmentation. This strategic selection and organization of datasets underscore u-LLaVA's commitment to advancing multi-modal task unification through Large Language Models.
</details> 

### **MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**

MoE-LLaVA introduces a novel approach by incorporating Mixture of Experts (MoE) within a large vision-language model, using learnable routers to selectively activate expert modules for processing specific tokens, thereby enhancing efficiency and enabling nuanced understanding of multimodal inputs.

[![arXiv](https://img.shields.io/badge/arXiv-2401.15947-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.15947) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/PKU-YuanGroup/MoE-LLaVA) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LanguageBind/MoE-LLaVA)  
Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Munan Ning, Li Yuan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5e214b-be64-4aac-aba4-04c97970b9de" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MoE-LLaVA**: Represents an innovative leap in the development of large vision-language models through the integration of **Mixture of Experts (MoE)** within a sophisticated architectural framework. This model is characterized by its sparse design, wherein individual tokens are directed towards a selection of experts based on **learnable routers**, ensuring that only the top-k experts are activated for any given token's processing. Such an approach not only enhances the model's efficiency but also its capability to handle diverse and complex data inputs by leveraging specialized processing paths for different types of information. At the heart of MoE-LLaVA's architecture are several critical components, including a vision encoder, **a visual projection MLP layer**, **word embedding layers**, **multi-head self-attention blocks**, **feed-forward neural networks**, and notably, **the MoE blocks** themselves. These elements are seamlessly integrated through the use of layer normalization and residual connections, establishing a robust and adaptable framework capable of deep multimodal understanding. The training methodology for MoE-LLaVA is meticulously structured in three stages, each designed to gradually enhance the model's proficiency in integrating and processing visual and textual data. This includes initial adaptation of image tokens, training of all LLM parameters excluding the vision encoder, and specialized training of the MoE layers, with the latter utilizing initialization weights from previous stages for optimal performance. Alignment techniques and fusion methods employed by MoE-LLaVA are pivotal in achieving a harmonious integration of text and image modalities. By utilizing learnable routers to dynamically allocate tokens to the most apt experts and subsequently processing these through a combination of LLM and MoE blocks, the model achieves a nuanced understanding of multimodal inputs. The datasets employed throughout the training phases—ranging from LLaVA-PT for pretraining to Hybrid-FT for multimodal instruction tuning, and LLaVA-FT for fine-tuning the MoE layers—further underscore the model's ability to refine its understanding across a broad spectrum of multimodal tasks. This strategic deployment of diverse datasets not only facilitates a comprehensive tuning of the model's capabilities but also underscores its potential in advancing the field of vision-language processing.
</details> 

### **BLIVA: A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions**

BLIVA augments the InstructBLIP model with a Visual Assistant, incorporating encoded patch embeddings alongside learned query embeddings to enhance the LLM's understanding of text-rich visual contexts, thereby excelling in handling complex visual questions.

[![arXiv](https://img.shields.io/badge/arXiv-2308.09936v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.09936v3) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mlpc-ucsd/bliva)  
Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/44c53b8a-ad35-4eca-a68b-63af32e6ccf1" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIVA**: This model builds upon the foundation of InstructBLIP, incorporating a Visual Assistant to enhance its understanding and processing of text-rich visual contexts. BLIVA's architecture is designed to capture the intricacies of visual content that may be overlooked during the query decoding process by melding learned query embeddings from InstructBLIP with directly projected encoded patch embeddings. The core components of BLIVA include a vision tower, responsible for encoding visual inputs into patch embeddings; a **Q-former**, which refines query embeddings; and a **projection layer** that bridges the visual and linguistic domains, enabling the LLM to access a rich tapestry of visual knowledge. The training methodology of BLIVA is structured around a two-stage scheme: initial pre-training on image-text pairs derived from captioning datasets, followed by instruction tuning using Visual Question Answering (VQA) data. This process begins with the pre-training of the projection layer for patch embeddings, succeeded by the fine-tuning of both the Q-former and the projection layer, while the image encoder and LLM remain static to prevent catastrophic forgetting. This approach ensures that BLIVA is finely attuned to visual information, enhancing its ability to handle complex visual questions. BLIVA's alignment techniques and fusion methods stand out for their integration of learned query embeddings with an additional visual assistant branch that utilizes encoded patch embeddings. By concatenating these embeddings and feeding them directly into the LLM, BLIVA significantly improves the model's text-image visual perception capabilities. This enhanced multimodal understanding is further demonstrated through the use of diverse datasets, including image captioning datasets for pre-training, instruction tuning VQA data for performance enhancement, and YTTB-VQA (YouTube Thumbnail Visual Question-Answer pairs) to showcase BLIVA's proficiency in processing text-rich images and its suitability for real-world applications.
</details> 

### **MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices**

MobileVLM offers a mobile-optimized vision-language model that combines a CLIP ViT-L/14 visual encoder with the efficient MobileLLaMA language model and a Lightweight Downsample Projector (LDP), enabling effective multimodal processing and alignment within the constraints of mobile devices.

[![arXiv](https://img.shields.io/badge/arXiv-2312.16886-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.16886) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/meituan-automl/mobilevlm)  
Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/59a06109-ba49-4299-951c-d7c0c562bca3" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MobileVLM**: Introduces a compact yet robust architecture designed to facilitate efficient vision-language tasks on mobile devices, distinguishing itself through a blend of specialized components and a streamlined training methodology tailored for edge computing environments. At its core, MobileVLM integrates a visual encoder based on the CLIP ViT-L/14 model with a resolution of 336x336, MobileLLaMA—a language model optimized for mobile devices, and a **Lightweight Downsample Projector (LDP)** that bridges the gap between visual and textual data with minimal computational overhead. This synergy between components ensures that MobileVLM can process and align multimodal inputs effectively, making it well-suited for mobile applications where resource efficiency is paramount. The training regimen for MobileVLM unfolds in three distinct phases, each contributing uniquely to the model's development. Initially, the language model undergoes pre-training using the text-centric RedPajama v1 dataset, laying a solid linguistic foundation. Subsequent supervised fine-tuning leverages multi-turn dialogues between humans and ChatGPT, refining the model's conversational abilities. The final stage involves training the integrated vision-language model on diverse multimodal datasets, equipping MobileVLM with the capacity to interpret and respond to both visual and textual stimuli. This comprehensive training approach ensures that MobileVLM achieves a balance between performance and efficiency, making it adept at handling complex vision-language interactions on mobile platforms. Central to MobileVLM's effectiveness is the Lightweight Downsample Projector (LDP), a novel component designed for the efficient alignment of visual and textual features. By employing mobile-friendly operations such as depth-wise convolution, LDP manages to downsample visual tokens to match the language model's input dimensions, preserving spatial information while minimizing computational demands. This alignment mechanism, in conjunction with the efficient fusion of vision and text embeddings, enables MobileVLM to maintain high levels of accuracy and responsiveness in mobile environments. Through the use of carefully selected datasets, including RedPajama v1 for linguistic pre-training and various multimodal datasets for comprehensive vision-language modeling, MobileVLM showcases its capability to navigate the challenges of mobile-based vision-language tasks with remarkable efficiency.
</details> 

### **FROZEN: Multimodal Few-Shot Learning with Frozen Language Models**

FROZEN enables multimodal few-shot learning by pairing a pre-trained, frozen language model with a trainable vision encoder (NF-ResNet-50) that converts images into a dynamic visual prefix, allowing the model to process and generate language in context with visual data without altering its core language capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2106.13884-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2106.13884)  
Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/4156475d-e501-495e-98bb-66efdd5b03f7" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FROZEN**: Presents an innovative approach to extending the few-shot learning capabilities of pre-existing language models into the multimodal domain, specifically targeting the integration of visual and linguistic elements without the need to alter the foundational language model parameters. This methodology introduces a vision encoder, specifically an **NF-ResNet-50**, designed to translate images into a continuous sequence of embeddings. These embeddings serve as a visual prefix to the input for a pre-trained autoregressive language model based on the Transformer architecture, enabling the language model to process and generate content relevant to the given visual context. The core innovation lies in the system's modularity, achieved by keeping the language model's weights static while **only updating the vision encoder** during training. This approach leverages the Conceptual Captions dataset, focusing on the alignment of image-caption pairs to train the vision encoder, thereby simplifying the integration of visual data into language models. The architecture of FROZEN is distinguished by its use of a dynamic visual prefix, a departure from the conventional static text prompts typical in prefix tuning. This dynamic prefix is achieved by linearly mapping and reshaping the vision encoder's output into a sequence of embeddings, mirroring the functionality of text-based prefix tokens in traditional language model tuning. This mechanism allows the model to adapt more fluidly to multimodal inputs, enhancing its ability to interpret and generate language that is contextually aligned with visual data. The employment of a dynamic visual prefix is a key factor in FROZEN's ability to improve task performance across multimodal settings through in-context learning, providing a novel solution to the challenge of incorporating visual information into the language generation process. The utilization of the Conceptual Captions dataset is central to FROZEN's training methodology, enabling the **vision encoder to adeptly convert images** into a format that the language model can process. This dataset serves the dual purpose of enhancing the model's understanding of visual content and its associated linguistic descriptions, thereby facilitating the generation of accurate and contextually relevant captions. The strategic combination of a static language model with a trainable vision encoder encapsulates FROZEN's approach to multimodal few-shot learning, offering a streamlined and effective pathway to integrating visual data into linguistic models.
</details> 

### **Flamingo: a Visual Language Model for Few-Shot Learning**

Flamingo pioneers a Perceiver-based VLM architecture that utilizes a Perceiver Resampler and gated cross-attention dense layers, enabling it to process interleaved text and visual sequences for impressive few-shot learning performance across a variety of multimodal tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2204.14198v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2204.14198v2)  
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/b46ebf3e-67fc-401e-a6ea-6f4797da372d" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Flamingo**: Represents an innovative approach in the realm of Visual Language Models (VLMs), specifically designed to excel in few-shot learning tasks. This model is distinguished by its capacity to process sequences of text tokens that are interwoven with visual data, such as images or videos, to generate textual outputs. At the core of Flamingo's architecture is the adoption of a Perceiver-based framework that adeptly manages high-resolution visual inputs. This design choice enables the handling of complex, multimodal information streams by transforming large visual feature maps into a concise number of visual tokens through the **Perceiver Resampler**. Further refining its architecture, Flamingo incorporates **gated cross-attention dense (GATED XATTN-DENSE) layers**, which play a pivotal role in conditioning the language model on visual inputs, thereby facilitating a nuanced understanding and generation of language based on the visual context. The training regimen of Flamingo is both extensive and diverse, encompassing a wide array of datasets culled from the web. This includes a rich mixture of interleaved image and text data, image-text pairs, and video-text pairs, which collectively contribute to the model's robust few-shot learning capabilities. A distinctive aspect of Flamingo's training is its strategy to minimize a weighted sum of per-dataset expected negative log-likelihoods of text given visual inputs. This approach, combined with a gradient accumulation strategy across all datasets, ensures comprehensive learning from varied multimodal contexts. The datasets employed in training, namely MultiModal MassiveWeb (M3W), ALIGN dataset, Long Text & Image Pairs (LTIP), and Video & Text Pairs (VTP), each serve a specific purpose. M3W facilitates training on interleaved text and image data, ALIGN on image-text pairs, LTIP on high-quality image-text pairs, and VTP on video-text pairs, ensuring Flamingo's adeptness across different visual language tasks. In its alignment techniques, Flamingo introduces an image-causal modeling approach to manage text-to-image cross-attention effectively, allowing the model to attend selectively to visual tokens of the image that immediately precede the given text token in the sequence. This capability is further enhanced by the gated cross-attention layers, which employ a tanh-gating mechanism to merge the output of these layers with the input representation from the residual connection. Such an alignment fusion method ensures that Flamingo can seamlessly integrate vision and text embeddings, underscoring its innovative architecture and the breadth of its training. Through these mechanisms, Flamingo stands out as a significant advancement in the integration of visual and textual data for language model training, showcasing its versatility and effectiveness in few-shot learning scenarios.
</details> 

### **OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models**

OpenFlamingo, an open-source adaptation of DeepMind's Flamingo, combines a CLIP ViT-L/14 visual encoder with a 7B parameter language model, utilizing frozen cross-attention modules for efficient and effective multimodal fusion during the decoding process, resulting in impressive performance on various vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2308.01390-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.01390) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mlfoundations/open_flamingo)  
Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**OpenFlamingo**: Represents an innovative leap in the integration of vision and language models, providing an open-source adaptation of DeepMind's Flamingo framework. This model is structured around a powerful combination of a CLIP Vision Transformer Large (ViT-L/14) for encoding visual inputs and a 7-billion parameter Multimodal Pretrained Transformer (MPT-7B) for language processing. The architecture is distinctive for its inclusion of cross-attention modules within every fourth decoder block of the language model, which remains frozen during training. These modules are pivotal for the model's ability to attentively merge visual information with textual context during the decoding process, thereby enhancing its multimodal understanding. The training methodology for OpenFlamingo is grounded in a comprehensive strategy that harnesses the vast data landscape of the internet. It utilizes a rich dataset amalgam comprising LAION-2B and the Multimodal version of the Common Crawl (C4) dataset, focusing on image-text pair sequences. This approach is facilitated by DistributedDataParallel training across an impressive array of 64 A100 80GB GPUs, leveraging automatic BF16 mixed precision for optimized performance. The model's alignment techniques are inspired by the original Flamingo's design philosophy, which emphasizes the importance of keeping the core vision and language models static while dynamically training the connecting **cross-attention modules** for decoding. This selective training process ensures that OpenFlamingo can effectively fuse visual and textual data, thereby significantly improving its proficiency in generating relevant text based on visual cues. Furthermore, the datasets used are instrumental in refining OpenFlamingo's capacity for understanding complex visual-textual interactions. Trained specifically on image-text sequences, the model demonstrates superior performance in tasks requiring nuanced interpretation of visual content, such as captioning, visual question answering, and image classification. This strategic focus on multimodal datasets underscores the model's purpose to bridge the gap between visual perception and linguistic expression, marking a substantial advancement in the field of multimodal AI. Through these architectural innovations and training strategies, OpenFlamingo sets a new standard for open-source models in the domain of visual-language tasks.
</details> 

### **IDEFICS**

IDEFICS, an 80B parameter vision-language model inspired by Flamingo, processes interleaved image and text sequences, utilizing a GPT-4 and Flamingo-based architecture to achieve robust multimodal understanding, trained on a diverse range of web-based datasets, including the specialized OBELICS dataset.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HuggingFaceM4/idefics-80b)
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**IDEFICS**: stands for "an 80 billion parameters vision and language model," distinguishing itself as a robust model designed to mimic Flamingo's capabilities while integrating substantial advancements in handling multimodal inputs. This model is crafted to accept sequences of images and text, generating text outputs that reflect a deep understanding of both visual and textual information. The architecture of IDEFICS builds on the foundations laid by GPT-4 and Flamingo, showcasing a harmonious blend of vision and language processing capabilities within a singular model framework. This strategic design allows IDEFICS to process and interpret complex multimodal inputs efficiently, setting a new precedent in the field of integrated vision-language models. During its development, IDEFICS faced challenges related to loss spikes, which were effectively mitigated through rollback strategies and precise adjustments in the learning rate. An auxiliary z-loss was introduced to normalize logits, significantly enhancing training stability. The model adopts Flamingo's methodological approach for alignment, utilizing pretrained vision and language backbones to foster a nuanced cross-modal understanding. Although specific details on fusion techniques for vision and text embeddings remain under wraps, it is inferred that the model employs **cross-attention mechanisms** akin to Flamingo's, facilitating a sophisticated integration of visual and textual data. Training on OBELICS—a meticulously curated collection of interleaved image-text web documents—and other web-scraped datasets, IDEFICS aims to excel in multimodal tasks. The OBELICS dataset, in particular, is designed to augment the model's performance by providing access to longer text contexts and a diverse array of web document types. This strategic dataset selection underscores IDEFICS's commitment to enhancing its proficiency across a spectrum of multimodal applications, leveraging the rich, varied content found in web documents to refine its understanding and output generation capabilities.
</details> 

### **PaLI: A Jointly-Scaled Multilingual Language-Image Model**

PaLI distinguishes itself as a jointly-scaled multilingual language-image model that utilizes a unified interface to process both unimodal and multimodal tasks, integrating a powerful ViT-e visual encoder with an mT5-based text encoder-decoder Transformer for comprehensive language and vision understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2209.06794-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2209.06794) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision)  
Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2565afb0-901c-4438-9488-c73a86261aa5" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PALI**: This model stands out by its ability to handle both unimodal (language or vision) and multimodal (language and vision together) tasks through a unified interface that accepts images and text as inputs, subsequently generating text as the output. The architecture of PALI ingeniously integrates a text encoder-decoder Transformer, based on pre-trained mT5 models, with visual tokens processed by a Vision Transformer (ViT) named ViT-e. ViT-e marks a significant advancement in visual processing with up to 4 billion parameters, setting a new precedent for the integration of visual components within language models. The PALI model utilizes pre-trained unimodal checkpoints, optimizing the efficiency of its training processes. Training methodologies for PALI are robust and diverse, incorporating a mixture of pre-training tasks aimed at enhancing the model's capability across a broad spectrum of downstream applications. Leveraging the expansive image-language dataset WebLI, which encompasses 10 billion images and texts across over 100 languages, PALI undergoes a comprehensive two-phase training regime. This includes a specific focus on high-resolution training for its largest model variant, PALI-17B. Such an approach ensures that PALI is not just multilingual but also highly adept at processing and understanding complex visual and textual data. The alignment and fusion techniques employed by PALI are particularly noteworthy. By adopting a unified modeling interface, the model treats various tasks with a task-agnostic perspective, allowing it to seamlessly transition between different types of vision and language tasks. The fusion of vision and text is achieved through **a cross-attention mechanism**, where a sequence of visual tokens from the Vision Transformer is integrated with the text encoder-decoder Transformer. This method enables an efficient and effective blending of multimodal information. The use of datasets such as WebLI, Conceptual Captions, and OCR data from WebLI, along with others like VQ2A-CC3M and Open Images, further enriches PALI's training, equipping it with a vast and versatile multimodal proficiency. This proficiency spans across multilingual settings, captioning, OCR, and visual question answering (VQA), ensuring PALI's comprehensive understanding and generation capabilities across a wide array of languages and tasks.
</details> 

### **PaLI-3 Vision Language Models: Smaller, Faster, Stronger**

PaLI-3 presents a powerful yet efficient vision-language model that integrates a contrastively pretrained 2B SigLIP vision model with a 3B UL2 Transformer, achieving impressive performance in tasks like captioning and visual question answering through a multi-stage training process that emphasizes scalability and robustness.

[![arXiv](https://img.shields.io/badge/arXiv-2310.09199-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.09199) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/kyegomez/PALI3)  
Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/92d34b30-b13b-44ed-90b5-3c8568a9b634" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PaLI-3** :Its architecture integrates a contrastively pretrained 2B **SigLIP vision model** with a 3B encoder-decoder UL2 Transformer, focusing on the efficient processing of visual and textual data. The training methodology of PaLI-3 includes **contrastive pretraining of the image encoder** on a vast scale of image-text data, subsequent multimodal training, and resolution increase stages to refine its performance further. These stages ensure that PaLI-3 achieves a nuanced understanding of visually-situated text and object localization, supported by datasets such as Web-scale image-text data, RefCOCO, WebLI, CC3M-35L, and various VQA datasets. The visual component of PaLI-3 utilizes a vision transformer pretrained in a contrastive manner, emphasizing efficiency, scalability, and robustness. This approach allows for a more nuanced pretraining of the image embedding component, which, when combined with text embeddings, enhances the model's ability to understand and generate text based on visual inputs. The full model employs these visual tokens alongside embedded input text tokens within a UL2 encoder-decoder framework, demonstrating its capability in generating text outputs for tasks such as captioning and visual question answering (VQA). PaLI-3's training process involves several key stages, starting with unimodal pretraining of the image encoder using image-text pairs from the web. This is followed by multimodal training, where the image encoder and text encoder-decoder are combined and trained on a mixture of tasks and data, focusing on visually-situated text and object detection. The resolution increase stage further enhances performance by fine-tuning the model with high-resolution inputs. Finally, task specialization involves fine-tuning PaLI-3 on individual benchmark tasks, optimizing its performance across a wide range of applications. 
</details> 

### **PaLM-E: An Embodied Multimodal Language Model**

PaLM-E innovates by embedding continuous sensory data, including images and sensor readings, into the language representation space of a pre-trained PaLM model, enabling it to process and generate text that reflects embodied reasoning and understanding of the physical world.

[![arXiv](https://img.shields.io/badge/arXiv-2303.03378-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.03378) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://palm-e.github.io)  
Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/67e5bbc7-1800-46e8-8ef1-b3b72a901a12" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PaLM-E**: Represents an innovative step in the development of multimodal language models by integrating continuous embodied observations—ranging from images and state estimates to various sensor modalities—into the linguistic embedding space of a pre-trained language model. It utilizes a decoder-only large language model (LLM) architecture that generates textual completions autoregressively, taking multimodal inputs into account. The core architecture of PaLM-E leverages a pre-trained PaLM as its language backbone, enhancing it with encoders that transform sensor modalities into a **sequence of vectors** compatible with the language model's embedding dimensions. This integration allows for the seamless combination of continuous sensor information with textual data, crafting multimodal sentences that the model processes. Training methodologies for PaLM-E are comprehensive and end-to-end, utilizing datasets composed of both continuous observations and textual information. The model employs a cross-entropy loss function for non-prefix tokens, with a training regimen that includes pre-trained Vision Transformers (ViTs) for image feature extraction alongside novel and pre-trained input encoders. The approach allows for flexibility in model training, including options for freezing pre-trained components or co-training them across varied data sets. This strategy ensures that PaLM-E benefits from both the depth of pre-trained models and the specificity of tailored encoders for continuous data. PaLM-E's alignment techniques and fusion methods are pivotal for its operation, employing encoders to integrate continuous sensor data into the linguistic embedding space effectively. This integration facilitates an understanding and generation of responses that reflect a blend of textual and sensor input, mimicking embodied reasoning. The model processes multimodal sentences—interleaved sequences of sensor observations and text—through its **self-attention layers**, similar to how it handles traditional text tokens. This methodology ensures a cohesive encoding of vision and text information. PaLM-E's training leverages a diverse array of datasets, including large-scale vision-and-language data and specialized robotics tasks datasets, aiming to excel across a broad spectrum of embodied reasoning tasks. This diverse training background enables PaLM-E to harness cross-domain transfer learning, enhancing its capabilities in specific robotics applications and general vision-language tasks alike.
</details> 

### **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**

MiniGPT-4 seamlessly blends visual and language processing by connecting a pretrained Vision Transformer and Q-Former to a frozen Vicuna LLM using a single linear projection layer, achieving impressive vision-language understanding through a two-stage training approach focused on efficient alignment and enhanced generation quality.

[![arXiv](https://img.shields.io/badge/arXiv-2304.10592v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.10592v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vision-cair/minigpt-4)  
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5ff945-1271-4189-8dd9-b0abd88eacc1" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MiniGPT-4**: presents an advanced integration of vision and language processing capabilities through a meticulously designed architecture that marries a frozen visual encoder with a frozen advanced Large Language Model (LLM), specifically Vicuna. At the heart of MiniGPT-4 is its novel approach to aligning visual and linguistic modalities: it employs **a single linear projection layer** to bridge the pretrained Vision Transformer (ViT) and **Q-Former** with the Vicuna LLM. This design choice underscores a commitment to efficiency, focusing on leveraging existing, robust components to achieve a seamless integration of visual features with sophisticated language capabilities. The training methodology for MiniGPT-4 is bifurcated into two distinct stages, optimizing both the initial alignment of visual and language features and the subsequent enhancement of generation reliability and naturalness. Initially, MiniGPT-4 undergoes training for 20,000 steps with a batch size of 256 on 4 A100 GPUs, utilizing a combined dataset from sources like Conceptual Captions, SBU, and LAION for foundational vision-language knowledge. This stage is crucial for establishing the basic alignment between the visual encoder and the Vicuna LLM. The second stage of finetuning, leveraging a curated dataset of 3,500 detailed image descriptions, is pivotal for refining the model's output, focusing on generating more detailed, reliable, and naturally flowing text. The strategic use of datasets in MiniGPT-4's training regimen underscores its dual objectives: foundational vision-language alignment and the enhancement of output naturalness and detail. Initial datasets facilitate the basic integration of visual and linguistic elements, while the curated dataset of detailed image descriptions serves to significantly improve the model's capability in generating nuanced and accurate natural language descriptions. Through this comprehensive and staged training approach, MiniGPT-4 achieves a refined balance between efficient visual-language alignment and the production of high-quality, detailed textual outputs, marking a significant step forward in the field of vision-language understanding.
</details> 

### **MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**

MiniGPT-v2 acts as a unified interface for vision-language multi-task learning by connecting a static Visual Transformer to a 7B parameter LLaMA-2-chat language model through a linear projection layer, efficiently processing high-resolution images and excelling in various tasks through a three-stage training approach.

[![arXiv](https://img.shields.io/badge/arXiv-2310.09478v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.09478v3)  
Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2354442a-0e96-4010-8b4f-8bc3d666427e" />
</p> 
<details>  
<summary>ℹ️ <i>More Information</i></summary>  
    
**MiniGPT-v2**: A sophisticated model designed to serve as a unified interface for vision-language multi-task learning, leveraging the innovative integration of a visual backbone with a large language model. At its core, the architecture combines a Visual Transformer (ViT) as its visual backbone, which is kept static during training, with **a linear projection layer** that effectively merges every four neighboring visual tokens into one. These consolidated tokens are then projected into the feature space of LLaMA-2-chat, a 7-billion parameter language model, facilitating the processing of high-resolution images (448x448 pixels). This structure allows MiniGPT-v2 to efficiently bridge the gap between visual input and language model processing, catering to a wide array of vision-language tasks. The training methodology employed by MiniGPT-v2 is particularly noteworthy, encompassing a three-stage strategy to comprehensively cover the spectrum of knowledge acquisition and task-specific performance enhancement. Initially, the model is exposed to a mix of weakly-labeled and fine-grained datasets, focusing on broad vision-language understanding. The training progressively shifts towards more fine-grained data to hone in on specific task improvements. In the final stage, MiniGPT-v2 is trained on multi-modal instruction and language datasets, aiming to refine its response to multi-modal instructions. The use of task-specific identifier tokens during training plays a crucial role in reducing ambiguity and sharpening task distinction, enabling the model to adeptly navigate the complexities of vision-language tasks. To support its extensive training and operational capabilities, MiniGPT-v2 utilizes a diverse array of datasets, including LAION, CC3M, SBU, GRIT-20M, COCO caption, and several others, each selected to fulfill distinct stages of the training process—from broad knowledge acquisition to task-specific improvements and sophisticated multi-modal instruction handling. This strategic dataset employment underscores MiniGPT-v2's capacity to assimilate and apply knowledge across a broad range of vision-language contexts, positioning it as a versatile tool in the evolving landscape of multi-task learning interfaces.
</details> 

### **LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents**

LLaVA-Plus pioneers the creation of multimodal agents by integrating diverse vision and vision-language models into a skill repository, enabling the agent to learn and use tools effectively through end-to-end training on comprehensive multimodal instruction-following data.

[![arXiv](https://img.shields.io/badge/arXiv-2311.05437-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.05437) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)  
Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, Chunyuan Li
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/1ede1c4f-bdeb-48e0-ae8e-ccfbee1dea51" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA-Plus**: Represents an innovative leap in the design of multimodal agents, integrating a diverse array of vision and vision-language pre-trained models into a comprehensive skill repository. This integration enables LLaVA-Plus to leverage end-to-end training to systematically expand its capabilities, allowing it to activate and combine relevant tools based on the users' multimodal inputs. The architecture of LLaVA-Plus is centered around a unified scheme for representing **multimodal instruction-following data**, which is essential for its advanced end-to-end trained multimodal instruction-following capabilities. The model is distinguished by its training methods, which utilize curated multimodal instruction-following data covering a broad spectrum of tasks, including visual understanding, generation, external knowledge retrieval, and their combinations. This approach allows LLaVA-Plus to incorporate new tools through instruction tuning, thereby expanding its abilities by learning to use these tools effectively. The training datasets—COCO, HierText, InfoSeek, JourneyDB, and Instruct P2P—are meticulously selected to enhance the model's training on visual understanding skills such as detection, segmentation, captioning, OCR, and external knowledge retrieval, alongside generation tasks and skill compositions. LLaVA-Plus employs unique alignment techniques and fusion methods that utilize raw visual signals during human-AI interaction sessions to improve tool use performance, planning, and reasoning. These techniques enable the seamless integration of vision and text embeddings by combining user inputs, tool activation prompts, and execution results into a unified dialogue format. This strategic approach not only facilitates enhanced interaction between the model and its users but also significantly boosts the model's overall performance and versatility in handling complex multimodal tasks.
</details> 

### **BakLLaVA**

BakLLaVA elevates the LLaVA framework by employing a Mistral 7B base enhanced with LLaVA 1.5 architecture, undergoing a meticulous two-stage training process on a diverse dataset to achieve superior performance in multimodal benchmarks, outperforming competitors like Llama 2 13B.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/skunkworksai/bakllava) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/SkunkworksAI/BakLLaVA-1)
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BakLLaVA**: Represents an innovative advancement in the realm of AI models, distinguishing itself with significant architectural enhancements over its predecessor, LLaVA. Developed with a strong focus on integrating multimodal capabilities into language models, BakLLaVA leverages a **Mistral 7B** base, augmented with the advanced **LLaVA 1.5 architecture**, to push the boundaries of performance in various benchmarks. This model has been meticulously designed to outperform notable predecessors, such as Llama 2 13B, across several benchmarks, showcasing the efficiency and effectiveness of its underlying architecture .The training methodology of BakLLaVA is particularly noteworthy, employing a feature alignment stage that utilizes 600K filtered CC3M images for establishing a robust vision-language connection. This process is complemented by a visual instruction tuning stage, where 150K GPT-generated multimodal instructions are utilized, signifying a tailored approach towards encoding vision and text together. Such a methodological approach not only enhances feature alignment but also optimizes the model for a broad spectrum of conceptual coverage, efficiency in training, and overall performance. BakLLaVA's architecture benefits from a diverse dataset compilation including 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP, alongside 158K GPT-generated multimodal instruction-following data, 450K academic-task-oriented VQA data, and 40K ShareGPT data, among others. This extensive dataset collection is pivotal for the model's training, ensuring broad concept coverage and reinforcing the model's capabilities in feature alignment and visual instruction tuning. The strategic selection of datasets underscores BakLLaVA's commitment to advancing AI's understanding and processing of complex visual and textual information, setting a new standard for multimodal AI models.
</details> 

### **CogVLM: Visual Expert for Pretrained Language Models**

CogVLM enhances pretrained language models with a dedicated visual expert module, incorporating a QKV matrix and MLP within each layer to achieve deep visual-language feature alignment, enabling superior performance in multimodal tasks such as image captioning and visual question answering.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03079v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03079v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/thudm/cogvlm)  
Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, Jie Tang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/93d951e1-ad49-47fd-9135-c11bc69d49bc" />
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CogVLM**: This approach enables the model to deeply fuse vision-language features, enhancing its ability to process and understand multimodal inputs. The architecture of CogVLM is built around several key components: a Vision Transformer (ViT) encoder, **an MLP adapter**, a pretrained large language model akin to GPT, and the innovative visual expert module. These components work in tandem to facilitate the model's advanced capabilities in handling complex visual and textual information. The training methodology for CogVLM is comprehensive, encompassing both pretraining and fine-tuning phases. During pretraining, the model undergoes learning with a focus on image captioning loss and Referring Expression Comprehension (REC) across an extensive dataset comprising over 1.5 billion image-text pairs and a visual grounding dataset featuring 40 million images. The fine-tuning phase employs a unified instruction-supervised approach across a variety of visual question-answering datasets, further refining the model's performance. CogVLM's alignment techniques are particularly noteworthy, employing **a visual expert module** in each layer that leverages a **QKV (Query, Key, Value) matrix** and an **MLP (Multilayer Perceptron)** to achieve deep visual-language feature alignment. This method not only allows for the seamless integration of image features into the language model's processing layers but also significantly enhances the model's overall multimodal processing capabilities. The datasets employed in training and refining CogVLM include LAION-2B, COYO-700M, a visual grounding dataset of 40 million images, and several visual question-answering datasets like VQAv2, OKVQA, TextVQA, OCRVQA, and ScienceQA. These datasets serve multiple purposes, from pretraining and instruction alignment to enhancing the model's proficiency in tasks such as image captioning and referring expression comprehension. Through this strategic use of diverse datasets, CogVLM is positioned to excel in a wide array of multimodal tasks, marking a significant advancement in the field of vision-language models.
</details> 

### **CogVLM2: Enhanced Vision-Language Models for Image and Video Understanding**

CogVLM2 is a family of open-source visual language models designed to push the boundaries of image and video understanding. This new generation builds upon the success of previous CogVLM models, focusing on enhanced vision-language fusion, efficient high-resolution architecture, and broader modalities and applications.

[![arXiv](https://img.shields.io/badge/arXiv-2408.16500-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.16500) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/THUDM/CogVLM2) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/THUDM/cogvlm2-6645f36a29948b67dc4eef75)  
Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, Lei Zhao, Zhuoyi Yang, Xiaotao Gu, Xiaohan Zhang, Guanyu Feng, Da Yin, Zihan Wang, Ji Qi, Xixuan Song, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Yuxiao Dong, Jie Tang  

<p align="center">
<img src="https://github.com/user-attachments/assets/f60247aa-66b3-486c-891c-c29cefe8aed4" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

CogVLM2 is a new generation visual language model designed for comprehensive image and video understanding. It leverages a powerful ViT encoder to extract visual features from high-resolution images or video sequences, which are then downsampled by a convolutional layer and aligned with linguistic representations through a SwiGLU module. This adapter efficiently bridges the visual and language modalities while preserving critical image information. The model then utilizes a visual expert architecture, integrating visual features into both the attention and FFN modules of the language decoder. This approach allows for deep vision-language fusion without compromising the model's inherent language capabilities. Notably, CogVLM2-Video extends this architecture to handle videos, incorporating timestamps alongside multi-frame inputs to enable temporal localization and question-answering capabilities. The CogVLM2 family has achieved state-of-the-art results on various benchmarks, including MMBench, MM-Vet, TextVQA, MVBench, and VCG-Bench, showcasing its versatility and effectiveness across a wide range of image and video understanding tasks.
</details>


### **Ferret: Refer and Ground Anything Anywhere at Any Granularity**

FERRET, a multimodal large language model, excels in spatial referencing and grounding by using a hybrid region representation that combines discrete coordinates with continuous features, allowing it to precisely pinpoint objects and regions within images, regardless of their complexity.

[![arXiv](https://img.shields.io/badge/arXiv-2310.07704v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.07704v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/apple/ml-ferret)  
Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, Yinfei Yang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/a5ff801f-d523-4383-8b89-e2499976b2bb" />
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FERRET**: stands as a multimodal large language model (MLLM) that pioneers in spatially referring to any object within an image, irrespective of its shape or granularity, and grounding open-vocabulary descriptions with precision. The architecture of FERRET is distinguished by its hybrid region representation, which marries discrete coordinates with continuous features to depict image regions. This novel approach enables the model to handle a wide range of spatial referring tasks, from pinpointing precise locations to addressing more abstract, shapeless areas within images. At the core of FERRET's architecture are several key components: an image encoder tasked with deriving image embeddings, **a spatial-aware visual sampler** designed to extract regional continuous features, and a language model that integrates image, text, and region features. This intricate setup facilitates the model's unique ability to understand and generate language that refers to spatial elements in images with unprecedented accuracy. The training of FERRET is conducted on the GRIT dataset, which includes over 1.1 million samples imbued with hierarchical spatial knowledge. This process is augmented by spatial-aware visual sampling techniques that cater to the diverse shapes and densities found in spatial data, allowing for the simultaneous generation of text and coordinates for objects within images.FERRET's alignment techniques and fusion methods are particularly noteworthy. By blending discrete coordinates with continuous visual features, the model can process inputs of freely formed regions and ground descriptions in its outputs accurately. This capability is supported by a diverse dataset portfolio, including GRIT for its rich spatial annotations, and Visual Genome, RefCOCOs, and Flickr30k for tasks such as object detection, phrase grounding, and evaluating the model's proficiency in referring and grounding. Through these methodologies, FERRET advances the field of multimodal language models by providing a versatile framework for spatial reasoning and language grounding in visual contexts.
</details> 

### **Fuyu-8B: A Multimodal Architecture for AI Agents**

Fuyu-8B introduces a streamlined architecture for AI agents by directly projecting image patches into a decoder-only transformer, simplifying multimodal processing by treating image and text tokens uniformly, and achieving efficient performance in vision-language tasks despite its straightforward design.

[![Link](https://img.shields.io/badge/https%3A%2F%2Fwww.adept.ai%2Fblog%2Ffuyu-8b?style=flat&label=Fuyu%208B
)](https://www.adept.ai/blog/fuyu-8b) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/adept/fuyu-8b)  
Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar

<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/61a75fb4-ced7-419c-bff7-7cb2e3ddc02d" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Fuyu-8B**: A streamlined multimodal model tailored for digital agents, distinguished by its unique approach to handling visual data and its integration with textual information. At the core of Fuyu-8B's architecture is a decoder-only transformer, a departure from traditional models that rely on separate image encoders. This design facilitates the direct projection of image patches into the transformer's initial layer with **a linear projection**, allowing Fuyu-8B to process images of any resolution without the need for complex training stages or the integration of resolution-specific mechanisms. The simplicity of this architecture does not only lie in its unified processing of image and text data but also in its elimination of the need for cross-attention mechanisms or adapters, streamlining the model's training and inference processes. In terms of alignment techniques, Fuyu-8B employs a novel approach by treating image tokens on par with text tokens from the inception of the model's processing pipeline. This method does away with separate position embeddings for images, thereby simplifying the alignment process between textual and visual data. The model's ability to support arbitrary image resolutions and perform fine-grained localization is particularly advantageous for applications requiring detailed visual understanding alongside textual interaction. The datasets utilized in Fuyu-8B's development, including VQAv2, OKVQA, COCO Captions, and AI2D, are instrumental in benchmarking the model against standard image understanding tasks such as visual question answering and caption generation. Despite Fuyu-8B's primary focus on applications within digital agents, the selection of these datasets ensures a comprehensive evaluation of its capabilities in broader contexts of image understanding and multimodal interaction. Through its innovative architecture and methodological simplicity, Fuyu-8B sets a new direction for the development of AI agents capable of sophisticated multimodal reasoning.
</details>

### **OtterHD: A High-Resolution Multi-modality Model**

OtterHD-8B, inspired by Fuyu-8B, directly integrates pixel-level information from high-resolution images (up to 1024x1024 pixels) into its language model using position embeddings, eliminating the need for a separate vision encoder and enabling precise interpretation of detailed visual inputs alongside textual instructions.

[![arXiv](https://img.shields.io/badge/arXiv-2311.04219v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.04219v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/luodian/otter) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Otter-AI/OtterHD-Demo)  
Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, Ziwei Liu
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**OtterHD-8B**: Represents an evolutionary step in multi-modality model design, building on the foundation of the **Fuyu-8B architecture** to interpret high-resolution visual inputs with exceptional precision. Unlike traditional models limited by fixed-size vision encoders, OtterHD-8B is equipped to handle flexible input dimensions, allowing for enhanced versatility across a variety of inference requirements. This model integrates pixel-level visual information directly into the language model without the need for a separate vision encoder, employing position embeddings to comprehend varying image sizes and enabling the processing of high-resolution images up to 1024x1024 pixels. Instruction tuning in OtterHD-8B is tailored towards accommodating various image resolutions, with the model being trained on a diverse dataset mixture including LLaVA-Instruct, VQAv2, GQA, OKVQA, OCRVQA, A-OKVQA, COCO-GOI, COCO-Caption, TextQA, RefCOCO, COCO-ITM, ImageNet, and LLaVA-RLHF. This training employs FlashAttention-2 and other fused operators for optimization, leveraging PyTorch and HuggingFace transformers. The direct integration of pixel-level information into the language model, facilitated by position embeddings, enables OtterHD-8B to understand and generate responses to high-resolution images alongside textual instructions without conventional vision and text embedding fusion methods. The datasets chosen for training OtterHD-8B underscore its focus on a broad array of vision and language tasks, including question answering, object recognition, and text-image alignment, aiming to enhance the model's capabilities in these areas. By directly processing image patches alongside textual instructions, OtterHD-8B eschews traditional fusion methods, leveraging its architecture to interpret and respond to complex multimodal inputs. This approach not only marks a significant advancement in handling high-resolution images but also in the model's overall ability to comprehend and interact with visual and textual data, positioning OtterHD-8B as a notable development in the field of multi-modality models.
</details>

### **SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models**

SPHINX pushes the boundaries of multi-modal LLMs by jointly mixing model weights, tasks, and visual embeddings during training, utilizing a two-stage approach that unfreezes the LLM (LLaMA-2) during pre-training for enhanced cross-modal learning and achieving impressive performance on a variety of vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2311.07575v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.07575v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alpha-vllm/)
Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, Yu Qiao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/3a1bf3fa-d0c5-4692-b9a8-97bea41ce226" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**SPHINX**: stands out as a multi-modal large language model (MLLM) designed to enhance the integration of language and vision through an innovative approach that includes the **joint mixing of model weights**, tuning tasks, and visual embeddings. This model is particularly distinguished by its methodology of unfreezing the large language model during pre-training to foster more effective cross-modal learning. The architecture of SPHINX is built upon a foundation that combines vision encoders, **two linear projection layers**, and leverages LLaMA-2 as the language model backbone. It adopts a two-stage training paradigm that emphasizes pre-training for vision-language alignment followed by fine-tuning aimed at visual instruction-following tasks. In the realm of training methodologies, SPHINX employs a strategy that emphasizes **the joint mixing of model weights**, tuning tasks, and visual embeddings, setting a precedent for robust cross-modal knowledge acquisition. This approach is complemented by a pre-training regimen that utilizes both real-world and synthetic data, thereby ensuring a comprehensive understanding across various visual instruction tasks. The model introduces an efficient strategy for processing high-resolution images, utilizing mixed scales and sub-images to accommodate diverse visual inputs. Moreover, SPHINX achieves vision-language alignment by integrating comprehensive visual embeddings, unfreezing the LLM during pre-training, and employing a weight-mixing strategy that bridges domain-specific knowledge across different network architectures and training paradigms. The datasets utilized in training SPHINX, including LAION-400M, LAION-COCO, RefinedWeb, VQAV2, GQA, OKVQA, A-OKVQA, OCRVQA, TextCaps, COCO, LVIS, RefCOCO, VG, and Flickr30k, serve a multifaceted purpose. They are instrumental in achieving multi-modal alignment, language-only tuning, and addressing a wide spectrum of visual question answering and general vision tasks. These tasks range from object detection and human pose estimation to referring object localization and understanding descriptions within the context of image regions. SPHINX, through its meticulous design and strategic training approach, sets a new benchmark in the field of multi-modal large language models, advancing the capabilities in vision-language integration.
</details>

### **CLIP: Contrastive Language-Image Pre-training**

CLIP leverages a contrastive learning approach, training separate image and text encoders on a massive dataset of 400 million image-text pairs to predict the most relevant captions for images, enabling impressive zero-shot transfer capabilities to various downstream tasks without requiring task-specific training data.

[![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2103.00020) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/openai/CLIP)  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c335c342-9a2c-4d4e-83d6-d3077cc32643" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CLIP**: model represents a groundbreaking approach in the field of machine learning, aiming to bridge the gap between visual and textual information through natural language supervision. Its architecture is designed to understand and predict **the most fitting captions for given images**, a methodology that stems from its training on a vast dataset of 400 million image-text pairs. This extensive training enables CLIP to learn state-of-the-art (SOTA) image representations and apply this knowledge to a wide range of downstream tasks without the need for task-specific training data, facilitating zero-shot transfer capabilities. At the core of CLIP are two primary components: **an image encoder** and **a text encoder**. These encoders are trained using a contrastive learning approach, optimizing for a contrastive objective that seeks to maximize the cosine similarity between correct image-text pairs while minimizing it for incorrect ones. This process is achieved through **a symmetric cross-entropy loss over the similarity scores between the embeddings of images and texts**, enabling the model to effectively link visual concepts with their linguistic descriptions. The model's ability to generalize across various tasks is further enhanced by its training methodology and the specific datasets it utilizes. By covering a broad spectrum of visual concepts and leveraging natural language for supervision, CLIP is adept at learning representations that are highly transferable to new tasks and domains. The custom dataset of 400 million image-text pairs, curated from the internet, plays a pivotal role in this process, providing the diverse and extensive visual and textual information necessary for the model to learn effectively. Through these innovations, CLIP sets a new standard for learning transferable visual models, showcasing the power of natural language in facilitating robust and versatile visual understanding.
</details> 

### **MetaCLIP: Demystifying CLIP Data**

MetaCLIP refines the data curation process for training vision-language models by employing algorithms that leverage CLIP-derived metadata to create a balanced and high-quality dataset from vast sources like CommonCrawl, resulting in improved performance and diversity compared to models trained on CLIP's original dataset.

[![arXiv](https://img.shields.io/badge/arXiv-2309.16671-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2309.16671) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/MetaCLIP)  
Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, Christoph Feichtenhofer
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/a6c79d0e-a4c7-48c9-86b6-3a8cc9853e11" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MetaCLIP**: Represents an innovative approach in the realm of data curation for machine learning, specifically targeting the **enhancement of training datasets** through metadata utilization derived from CLIP's concepts. This model is designed to sift through extensive raw data pools, such as the CommonCrawl dataset, to curate a high-quality, balanced subset that significantly betters the diversity and performance metrics of the data used for training machine learning models. The essence of MetaCLIP lies in its unique architecture that incorporates data curation algorithms, which are adept at leveraging metadata for the purpose of balancing and enriching the training dataset both in terms of quality and diversity. The architecture of MetaCLIP is structured around these **data curation algorithms**, which play a pivotal role in the framework by identifying and assembling a balanced and high-quality dataset from a vast collection of 400 million image-text pairs initially sourced from CommonCrawl. This process is instrumental in MetaCLIP's ability to demonstrate superior performance on various benchmarks, including zero-shot ImageNet classification, when compared to datasets curated using CLIP's original methodologies. The training methods employed by MetaCLIP, therefore, are not just about processing and learning from data but also about intelligently selecting the data that is most beneficial for the training process, ensuring that the model is trained on a dataset that is representative, diverse, and of high quality. The purpose behind employing datasets like CommonCrawl within the MetaCLIP framework is to address and overcome the limitations observed in CLIP's original dataset. By curating a balanced and high-quality dataset of 400 million image-text pairs, MetaCLIP sets a new precedent in the field of machine learning data curation. This strategic selection and enhancement of the training dataset enable MetaCLIP to significantly improve performance on standard benchmarks compared to its predecessor, highlighting the importance of dataset quality and diversity in achieving high performance in machine learning tasks. Through its innovative approach to data curation, MetaCLIP offers a promising avenue for enhancing the capabilities of machine learning models, particularly in applications requiring robust image-text understanding and classification.
</details> 

### **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**

Alpha-CLIP builds upon the CLIP model by incorporating region awareness through the addition of an alpha channel to the image encoder, trained on millions of RGBA region-text pairs, enabling precise control over image emphasis and enhancing performance across various tasks requiring detailed spatial understanding.

[![arXiv](https://img.shields.io/badge/arXiv-22312.03818-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.03818) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/SunzeY/AlphaCLIP)  
Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, Jiaqi Wang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/07bd6161-1682-4954-97f3-3770258bfa8c" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**Alpha-CLIP**: Introduces a significant enhancement to the original CLIP model, incorporating region awareness to its repertoire of capabilities. This model is fine-tuned on millions of RGBA region-text pairs, enabling it to maintain CLIP's visual recognition prowess while offering precise control over the emphasis of image content. By integrating an additional **alpha channel into the CLIP image encoder**, Alpha-CLIP allows for detailed segmentation and region-specific processing without modifying the foundational CLIP weights, thus facilitating a nuanced approach to image understanding that respects the spatial dynamics of visual data.
The training of Alpha-CLIP leverages a novel data generation pipeline designed to produce a vast array of RGBA-region text pairs. This process involves the creation of natural images equipped with foreground alpha channels and their corresponding referring expressions for specific regions. Such a methodology not only enables the fine-tuning of the model with an additional alpha channel input but also underpins its ability to perform with heightened specificity across various tasks. These tasks range from image recognition to multimodal large language models, and extend into both 2D and 3D generation domains, showcasing Alpha-CLIP's versatility and broad applicability. Datasets like LAION-400M, LAION-5B, and GRIT play a crucial role in training Alpha-CLIP, providing a wide spectrum of images for initial training and fine-grained mask-level labels for enhancing local perception capabilities. This strategic choice of datasets ensures that Alpha-CLIP is not only well-equipped for general visual recognition tasks but also capable of nuanced, region-specific processing and understanding, setting a new standard for models at the intersection of language and vision.
</details> 

### **GLIP: Grounded Language-Image Pre-training**

GLIP revolutionizes language-image pre-training by unifying object detection and phrase grounding, allowing it to understand and execute tasks requiring object-level precision and language awareness through a deep integration of visual and textual information during training.

[![arXiv](https://img.shields.io/badge/arXiv-2112.03857-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2112.03857) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/GLIP)  
Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/06e6f8dc-fbd8-49da-8651-a22ee2edcf3d" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**GLIP**: A novel approach that innovatively unifies the tasks of object detection and phrase grounding by redefining object detection as a phrase grounding challenge. This strategic reformation allows the model to exploit extensive image-text paired datasets for pre-training, equipping it with the capability to comprehend and execute tasks that require object-level precision, language awareness, and semantically rich visual representations. At its core, GLIP's architecture is designed to deeply integrate visual and textual information, enhancing its understanding of complex visual scenes in conjunction with textual prompts. The architecture of GLIP is composed of several critical components, including a visual encoder that can either be a Convolutional Neural Network (CNN) or a Transformer, tasked with extracting features from regions or bounding boxes within images. It also includes a language encoder dedicated to processing text prompts and prediction heads (box classifier and box regressor) that are trained using **classification** and **localization loss**. A distinctive feature of GLIP is its method of deep fusion between image and text, specifically in the latter stages of encoding, which merges visual and textual information more comprehensively than traditional methods. GLIP's training methodology is as innovative as its architecture, employing a unified formulation that amalgamates detection and grounding tasks into a singular workflow. This model is trained end-to-end, optimizing losses defined for **both detection** (focusing on localization and classification) and **grounding** (centering on alignment scores between image regions and corresponding words in the prompt). Such deep integration of visual and language features during training is pivotal, facilitating the model's ability to learn effectively from paired image-text data. The datasets utilized for training GLIP, including COCO, OpenImages, Objects365, Visual Genome, Flickr30k-entities, LVIS, and PhraseCut, are meticulously selected to cover a wide array of object classes and scenarios, each serving a unique purpose from object detection and phrase grounding to instance segmentation and referring expression segmentation. Through this comprehensive training, GLIP sets a new precedent in the realm of language-image pre-training, demonstrating advanced capabilities in interpreting and interacting with both visual and textual data.
</details>

### **ImageBind: One Embedding Space To Bind Them All**

ImageBind revolutionizes multimodal learning by creating a single, joint embedding space that integrates six modalities – images, text, audio, depth, thermal, and IMU data – through image-paired data as a central binding agent, allowing for zero-shot classification and retrieval across diverse data types.

[![arXiv](https://img.shields.io/badge/arXiv-2305.05665-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.05665) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/imagebind)  
Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/fbf8bcdd-b1bb-4fd8-8723-3c82e84ef759" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**ImageBind**: Introduces an innovative approach to multimodal learning by creating **a joint embedding space** that encompasses six different modalities: **images, text, audio, depth, thermal, and IMU (Inertial Measurement Unit)** data. This model uniquely employs image-paired data as a central binding agent, enabling it to leverage the capabilities of large-scale vision-language models to extend zero-shot capabilities to new, previously unlinked modalities. By doing so, ImageBind not only facilitates a deeper integration of diverse data types but also opens up new avenues for zero-shot classification and retrieval across a wide range of applications. At the heart of ImageBind's architecture lies a transformer-based design, adapted for each specific modality to ensure optimal processing and representation. For instance, it utilizes a Vision Transformer for image data, with each modality encoder being augmented by **modality-specific linear projection heads**. These adaptations are crucial for maintaining a uniform embedding size across the disparate data types, ensuring that the model can effectively learn from and link together the various modalities. This uniformity is key to ImageBind's ability to create a cohesive and comprehensive embedding space that captures the nuances of each data type. The training methodology behind ImageBind is particularly noteworthy. It employs contrastive learning, utilizing both web-scale image-text data and naturally occurring paired data from various modalities, such as video-audio and image-depth pairs. This strategy allows the model to learn a single joint embedding space without requiring all modalities to co-occur, a significant advantage that enhances its flexibility and applicability. The use of datasets like Audioset, SUN RGB-D, LLVIP, and Ego4D, which provide naturally paired data across the model's target modalities, is critical to this process. These datasets enable ImageBind to achieve emergent zero-shot classification and retrieval performance on tasks tailored to each modality, showcasing the model's ability to seamlessly navigate and leverage the complex interplay between different forms of data.
</details>

### **SigLIP: Sigmoid Loss for Language Image Pre-Training**

SigLIP introduces a simple pairwise sigmoid loss for language-image pre-training, allowing for scalable training with large batch sizes without compromising performance, enabling efficient alignment between image and text representations.

[![arXiv](https://img.shields.io/badge/arXiv-2303.15343-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.15343)  
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/60018313-37dd-4dbd-8eb4-a3075fd26663" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**SigLIP**: A novel approach to language-image pre-training by proposing **a simple pairwise sigmoid loss**. This method contrasts with standard contrastive learning that utilizes softmax normalization, as it operates directly on image-text pairs without necessitating a global view of pairwise similarities for normalization. The primary advantage of this approach is its scalability, allowing for the use of larger batch sizes without compromising performance. The architecture leverages a vision transformer for image processing and a conventional transformer for text, with the sigmoid loss facilitating independent processing of image-text pairs. This design enables more efficient training dynamics, particularly in the context of large batch sizes, by examining the effects of varying the negative to positive ratio and the selection of example pairs. Training methodologies focus on exploiting large batch sizes, delving into the dynamics of how batch size variations influence model performance. The introduction of sigmoid loss is pivotal, enabling the model to train effectively with these large batches by investigating the relationship between the ratio of negative to positive examples and the optimization of example pair selection. The use of the LiT image-text dataset and the WebLI dataset is integral to the model's training, aiming to achieve aligned representational spaces between images and texts. These datasets are chosen for their utility in assessing zero-shot transfer capabilities, as well as in exploring the scalability and efficiency of the model's sigmoid loss-based training. In essence, SigLIP marks a significant stride in language-image pre-training through its innovative use of sigmoid loss, enhancing scalability and training efficiency. This approach not only simplifies the training process by eliminating the need for global normalization but also showcases the model's adaptability to large-scale data handling. The strategic selection of datasets further underscores the model's capability to forge aligned representational spaces, paving the way for advanced zero-shot learning and efficient multimodal integration.
</details>

### **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

The Vision Transformer (ViT) revolutionizes image recognition by applying the Transformer architecture to images, processing them as a sequence of fixed-size patches, thereby demonstrating that image recognition can benefit from the power of transformers, surpassing traditional convolutional neural network (CNN) approaches with the aid of large-scale training datasets.

[![arXiv](https://img.shields.io/badge/arXiv-2010.11929v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2010.11929v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/vision_transformer)  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/b2f77966-c2e8-4204-ba90-be51196a7dee" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**The Vision Transformer (ViT)**: A paradigm shift in image recognition by applying the transformer architecture, predominantly used in natural language processing, directly to images. It innovatively processes images as **a sequence of fixed-size patches**, akin to how tokens are treated in **text applications**. This approach is facilitated through minimal modifications to the standard transformer components, emphasizing the model's adaptability to visual tasks without relying on the convolutional neural networks' (CNNs) inductive biases. ViT's architecture is distinguished by its use of linear embedding for **image patches** and **position embeddings**, which are crucial for maintaining the spatial hierarchy of image data. The core of ViT is a standard Transformer encoder that includes multiheaded self-attention (MSA) and multilayer perceptron (MLP) blocks, complemented by layer normalization and residual connections, underscoring its efficiency and robustness in handling visual data. Training methodologies for ViT are characterized by its scalability and the significant impact of dataset size on its performance. Initially, ViT exhibits modest accuracies without strong regularization techniques. However, its performance escalates with the scale of training, showcasing its potential to outperform traditional CNN approaches through extensive pre-training on large datasets. This process highlights the critical role of dataset selection in ViT's training regimen. It is fine-tuned on smaller datasets following a comprehensive pre-training phase that leverages large datasets like ImageNet-21k and JFT-300M to enhance model generalization and performance across a wide range of tasks. The datasets employed, including ImageNet, CIFAR-100, VTAB, ImageNet-21k, and JFT-300M, serve dual purposes: benchmarking the model's image classification capabilities and evaluating its transferability to diverse tasks with limited data, thereby establishing ViT's versatility and effectiveness in advancing image recognition tasks.
</details>

</details> 
 <p align="center">
</p>
<details> 
<summary>ℹ️ <i>垂类VLM（中文）</i></summary>
 
</h1>
<div align="center">
    <h1>Awesome Chinese LLM</h1>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"/></a>
</div>

<p align="center" width="100%">
<img src="src/icon.png" alt="Awesome-Chinese-LLM" style="width: 20%; height: auto; display: inline-block; margin: auto; border-radius: 50%;">
</p>
<p align="center">
<font face="黑体" color=orange size=5"> An Awesome Collection for LLM in Chinese </font>
</p>
<p align="center">
<font face="黑体" color=orange size=5"> 收集和梳理中文LLM相关 </font>
</p>
<p align="center">
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/stargazers"> <img src="https://img.shields.io/github/stars/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub stars"></a>
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/issues"> <img src="https://img.shields.io/github/issues/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub issues"></a>
  <a href="https://github.com/HqWu-HITCS/Awesome-Chinese-LLM/forks"> <img src="https://img.shields.io/github/forks/HqWu-HITCS/Awesome-Chinese-LLM.svg?style=popout-square" alt="GitHub forks"></a>
</p>

自ChatGPT为代表的大语言模型（Large Language Model, LLM）出现以后，由于其惊人的类通用人工智能（AGI）的能力，掀起了新一轮自然语言处理领域的研究和应用的浪潮。尤其是以ChatGLM、LLaMA等平民玩家都能跑起来的较小规模的LLM开源之后，业界涌现了非常多基于LLM的二次微调或应用的案例。本项目旨在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料，目前收录的资源已达100+个！

如果本项目能给您带来一点点帮助，麻烦点个⭐️吧～

同时也欢迎大家贡献本项目未收录的开源模型、应用、数据集等。提供新的仓库信息请发起PR，并按照本项目的格式提供仓库链接、star数，简介等相关信息，感谢~

![Awesome-Chinese-LLM](src/LLM.png)

常见底座模型细节概览：
| 底座     | 包含模型                    | 模型参数大小      | 训练token数  | 训练最大长度 | 是否可商用 |
|----------|---------------------------|-----------------|-------------|------------|-------   |
| ChatGLM  | ChatGLM/2/3/4 Base&Chat   | 6B              | 1T/1.4      | 2K/32K     | 可商用   |
| LLaMA    | LLaMA/2/3 Base&Chat       | 7B/8B/13B/33B/70B | 1T/2T       | 2k/4k      | 部分可商用  |
| Baichuan | Baichuan/2 Base&Chat      | 7B/13B          | 1.2T/1.4T | 4k     | 可商用   |
| Qwen     | Qwen/1.5/2/2.5 Base&Chat&VL   | 7B/14B/32B/72B/110B | 2.2T/3T/18T      | 8k/32k     | 可商用   |
| BLOOM    | BLOOM                     | 1B/7B/176B-MT   | 1.5T      | 2k     | 可商用   |
| Aquila   | Aquila/2 Base/Chat        | 7B/34B          | -         | 2k     | 可商用   |
| InternLM | InternLM/2/2.5 Base/Chat/VL   | 7B/20B          | -         | 200k | 可商用 |
| Mixtral  | Base&Chat                 | 8x7B            | -         | 32k | 可商用 |
| Yi       | Base&Chat                 | 6B/9B/34B       | 3T        | 200k | 可商用 |
| DeepSeek | Base&Chat                 | 1.3B/7B/33B/67B | -         | 4k | 可商用 |
| XVERSE   | Base&Chat                 | 7B/13B/65B/A4.2B| 2.6T/3.2T | 8k/16k/256k | 可商用 |

## 目录

- [目录](#目录)
  - [1. 模型](#1-模型)
    - [1.1 文本LLM模型](#11-文本llm模型)
    - [1.2 多模态LLM模型](#12-多模态llm模型)
  - [2. 应用](#2-应用)
    - [2.1 垂直领域微调](#21-垂直领域微调)
      - [医疗](#医疗)
      - [法律](#法律)
      - [金融](#金融)
      - [教育](#教育)
      - [科技](#科技)
      - [电商](#电商)
      - [网络安全](#网络安全)
      - [农业](#农业)
    - [2.2 LangChain应用](#22-langchain应用)
    - [2.3 其他应用](#23-其他应用)
  - [3. 数据集](#3-数据集)
    - [预训练数据集](#预训练数据集)
    - [SFT数据集](#sft数据集)
    - [偏好数据集](#偏好数据集)
  - [4. LLM训练微调框架](#4-llm训练微调框架)
  - [5. LLM推理部署框架](#5-llm推理部署框架)
  - [6. LLM评测](#6-llm评测)
  - [7. LLM教程](#7-llm教程)
    - [LLM基础知识](#llm基础知识)
    - [提示工程教程](#提示工程教程)
    - [LLM应用教程](#llm应用教程)
    - [LLM实战教程](#llm实战教程)
  - [8. 相关仓库](#8-相关仓库)
- [Star History](#star-history)

### 1. <a name='模型'></a>模型

#### 1.1 文本LLM模型

* ChatGLM：
  * 地址：https://github.com/THUDM/ChatGLM-6B
    ![](https://img.shields.io/github/stars/THUDM/ChatGLM-6B.svg)
  * 简介：中文领域效果最好的开源底座模型之一，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持
* ChatGLM2-6B
  * 地址：https://github.com/THUDM/ChatGLM2-6B
    ![](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg)
  * 简介：基于开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，引入了GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练；基座模型的上下文长度扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练；基于 Multi-Query Attention 技术实现更高效的推理速度和更低的显存占用；允许商业使用。
* ChatGLM3-6B
  * 地址：https://github.com/THUDM/ChatGLM3
    ![](https://img.shields.io/github/stars/THUDM/ChatGLM3.svg)
  * 简介：ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型： ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略；更完整的功能支持： ChatGLM3-6B 采用了全新设计的 Prompt 格式，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景；更全面的开源序列： 除了对话模型 ChatGLM3-6B 外，还开源了基础模型 ChatGLM3-6B-Base、长文本对话模型 ChatGLM3-6B-32K。以上所有权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用。
* GLM-4
  * 地址：https://github.com/THUDM/GLM-4
    ![](https://img.shields.io/github/stars/THUDM/GLM-4.svg)
  * 简介：GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中， **GLM-4-9B** 及其人类偏好对齐的版本 **GLM-4-9B-Chat** 均表现出超越 Llama-3-8B 的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的 **GLM-4-9B-Chat-1M** 模型和基于 GLM-4-9B 的多模态模型 GLM-4V-9B。**GLM-4V-9B** 具备 1120 * 1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B 表现出超越 GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。
* Qwen/Qwen1.5/Qwen2/Qwen2.5
  * 地址：https://github.com/QwenLM
    ![](https://img.shields.io/github/stars/QwenLM/Qwen.svg)
  * 简介：通义千问 是阿里云研发的通义千问大模型系列模型，包括参数规模为18亿（1.8B）、70亿（7B）、140亿（14B）、720亿（72B）和1100亿（110B）。各个规模的模型包括基础模型Qwen，以及对话模型。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域，能支持8~32K的上下文长度，针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。
* InternLM
  * 地址：https://github.com/InternLM/InternLM-techreport
    ![](https://img.shields.io/github/stars/InternLM/InternLM-techreport.svg)
  * 简介：商汤科技、上海AI实验室联合香港中文大学、复旦大学和上海交通大学发布千亿级参数大语言模型“书生·浦语”（InternLM）。据悉，“书生·浦语”具有1040亿参数，基于“包含1.6万亿token的多语种高质量数据集”训练而成。
* InternLM2
  * 地址：https://github.com/InternLM/InternLM
      ![](https://img.shields.io/github/stars/InternLM/InternLM.svg)
  * 简介：商汤科技、上海AI实验室联合香港中文大学、复旦大学和上海交通大学发布千亿级参数大语言模型“书生·浦语”（InternLM2）。InternLM2 在数理、代码、对话、创作等各方面能力都获得了长足进步，综合性能达到开源模型的领先水平。InternLM2 包含两种模型规格：7B 和 20B。7B 为轻量级的研究和应用提供了一个轻便但性能不俗的模型，20B 模型的综合性能更为强劲，可以有效支持更加复杂的实用场景。
* DeepSeek-V2
  * 地址：https://github.com/deepseek-ai/DeepSeek-V2
    ![](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-V2.svg)
  * 简介：DeepSeek-V2：强大、经济、高效的专家混合语言模型
* Baichuan-7B
  * 地址：https://github.com/baichuan-inc/Baichuan-7B
    ![](https://img.shields.io/github/stars/baichuan-inc/baichuan-7B.svg)
  * 简介：由百川智能开发的一个开源可商用的大规模预训练语言模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。
* Baichuan-13B
  * 地址：https://github.com/baichuan-inc/baichuan-13B
    ![](https://img.shields.io/github/stars/baichuan-inc/baichuan-13B.svg)
  * 简介：Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。该项目发布包含有预训练 (Baichuan-13B-Base) 和对齐 (Baichuan-13B-Chat) 两个版本。
* Baichuan2
  * 地址：https://github.com/baichuan-inc/Baichuan2
    ![](https://img.shields.io/github/stars/baichuan-inc/Baichuan2.svg)
  * 简介：由百川智能推出的新一代开源大语言模型，采用 2.6 万亿 Tokens 的高质量语料训练，在多个权威的中文、英文和多语言的通用、领域 benchmark上取得同尺寸最佳的效果，发布包含有7B、13B的Base和经过PPO训练的Chat版本，并提供了Chat版本的4bits量化。
* XVERSE-7B
  * 地址：https://github.com/xverse-ai/XVERSE-7B
    ![](https://img.shields.io/github/stars/xverse-ai/XVERSE-7B.svg)
  * 简介：由深圳元象科技自主研发的支持多语言的大语言模型，支持 8K 的上下文长度（Context Length），使用 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，支持中、英、俄、西等 40 多种语言。并包含GGUF、GPTQ量化版本的模型，支持在llama.cpp、vLLM在MacOS/Linux/Windows系统上推理。
* XVERSE-13B
  * 地址：https://github.com/xverse-ai/XVERSE-13B
    ![](https://img.shields.io/github/stars/xverse-ai/XVERSE-13B.svg)
  * 简介：由深圳元象科技自主研发的支持多语言的大语言模型，支持 8K 的上下文长度（Context Length），使用 3.2 万亿 token 的高质量、多样化的数据对模型进行充分训练，支持中、英、俄、西等 40 多种语言。包含长序列对话模型 XVERSE-13B-256K ，该版本模型最大支持 256K 的上下文窗口长度，约 25w 字的输入内容，可以协助进行文献总结、报告分析等任务。并包含GGUF、GPTQ量化版本的模型，支持在llama.cpp、vLLM在MacOS/Linux/Windows系统上推理。
* XVERSE-65B
  * 地址：https://github.com/xverse-ai/XVERSE-65B
    ![](https://img.shields.io/github/stars/xverse-ai/XVERSE-65B.svg)
  * 简介：由深圳元象科技自主研发的支持多语言的大语言模型，支持 16K 的上下文长度（Context Length），使用 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，支持中、英、俄、西等 40 多种语言。包含增量预训练到 3.2 万亿 token 的 XVERSE-65B-2 模型。并包含GGUF、GPTQ量化版本的模型，支持在llama.cpp、vLLM在MacOS/Linux/Windows系统上推理。
* XVERSE-MoE-A4.2B
  * 地址：https://github.com/xverse-ai/XVERSE-MoE-A4.2B
    ![](https://img.shields.io/github/stars/xverse-ai/XVERSE-MoE-A4.2B.svg)
  * 简介：由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），使用混合专家模型（MoE，Mixture-of-experts）架构，模型的总参数规模为 258 亿，实际激活的参数量为 42 亿，支持 8K 的上下文长度（Context Length），使用 3.2 万亿 token 的高质量、多样化的数据对模型进行充分训练，支持中、英、俄、西等 40 多种语言。
* Skywork
  * 地址：https://github.com/SkyworkAI/Skywork
    ![](https://img.shields.io/github/stars/SkyworkAI/Skywork.svg)
  * 简介：该项目开源了天工系列模型，该系列模型在3.2TB高质量多语言和代码数据上进行预训练，开源了包括模型参数，训练数据，评估数据，评估方法。具体包括Skywork-13B-Base模型、Skywork-13B-Chat模型、Skywork-13B-Math模型和Skywork-13B-MM模型，以及每个模型的量化版模型，以支持用户在消费级显卡进行部署和推理。
* Yi
  * 地址：https://github.com/01-ai/Yi
    ![](https://img.shields.io/github/stars/01-ai/Yi.svg)
  * 简介：该项目开源了Yi-6B和Yi-34B等模型，该系列模型最长可支持200K的超长上下文窗口版本，可以处理约40万汉字超长文本输入，理解超过1000页的PDF文档。
* Chinese-LLaMA-Alpaca：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
    ![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg)
  * 简介：中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署，在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练
* Chinese-LLaMA-Alpaca-2：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
    ![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca-2.svg)
  * 简介：该项目将发布中文LLaMA-2 & Alpaca-2大语言模型，基于可商用的LLaMA-2进行二次开发。
* Chinese-LlaMA2：
  * 地址：https://github.com/michael-wzhu/Chinese-LlaMA2
    ![](https://img.shields.io/github/stars/michael-wzhu/Chinese-LlaMA2.svg)
  * 简介：该项目基于可商用的LLaMA-2进行二次开发决定在次开展Llama 2的中文汉化工作，包括Chinese-LlaMA2: 对Llama 2进行中文预训练；第一步：先在42G中文预料上进行训练；后续将会加大训练规模；Chinese-LlaMA2-chat: 对Chinese-LlaMA2进行指令微调和多轮对话微调，以适应各种应用场景和多轮对话交互。同时我们也考虑更为快速的中文适配方案：Chinese-LlaMA2-sft-v0: 采用现有的开源中文指令微调或者是对话数据，对LlaMA-2进行直接微调 (将于近期开源)。
* Llama2-Chinese：
  * 地址：https://github.com/FlagAlpha/Llama2-Chinese
    ![](https://img.shields.io/github/stars/FlagAlpha/Llama2-Chinese.svg)
  * 简介：该项目专注于Llama2模型在中文方面的优化和上层建设，基于大规模中文数据，从预训练开始对Llama2模型进行中文能力的持续迭代升级。
* OpenChineseLLaMA：
  * 地址：https://github.com/OpenLMLab/OpenChineseLLaMA
    ![](https://img.shields.io/github/stars/OpenLMLab/OpenChineseLLaMA.svg)
  * 简介：基于 LLaMA-7B 经过中文数据集增量预训练产生的中文大语言模型基座，对比原版 LLaMA，该模型在中文理解能力和生成能力方面均获得较大提升，在众多下游任务中均取得了突出的成绩。
* BELLE：
  * 地址：https://github.com/LianjiaTech/BELLE
    ![](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg)
  * 简介：开源了基于BLOOMZ和LLaMA优化后的一系列模型，同时包括训练数据、相关模型、训练代码、应用场景等，也会持续评估不同训练数据、训练算法等对模型表现的影响。
* Panda：
  * 地址：https://github.com/dandelionsllm/pandallm
    ![](https://img.shields.io/github/stars/dandelionsllm/pandallm.svg)
  * 简介：开源了基于LLaMA-7B, -13B, -33B, -65B 进行中文领域上的持续预训练的语言模型, 使用了接近 15M 条数据进行二次预训练。
* Robin (罗宾):
  * 地址：https://github.com/OptimalScale/LMFlow
    ![](https://img.shields.io/github/stars/OptimalScale/LMFlow.svg)
  * 简介：Robin (罗宾)是香港科技大学LMFlow团队开发的中英双语大语言模型。仅使用180K条数据微调得到的Robin第二代模型，在Huggingface榜单上达到了第一名的成绩。LMFlow支持用户快速训练个性化模型，仅需单张3090和5个小时即可微调70亿参数定制化模型。
* Fengshenbang-LM：
  * 地址：https://github.com/IDEA-CCNL/Fengshenbang-LM
    ![](https://img.shields.io/github/stars/IDEA-CCNL/Fengshenbang-LM.svg)
  * 简介：Fengshenbang-LM(封神榜大模型)是IDEA研究院认知计算与自然语言研究中心主导的大模型开源体系，该项目开源了姜子牙通用大模型V1，是基于LLaMa的130亿参数的大规模预训练模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。除姜子牙系列模型之外，该项目还开源了太乙、二郎神系列等模型。
* BiLLa：
  * 地址：https://github.com/Neutralzz/BiLLa
    ![](https://img.shields.io/github/stars/Neutralzz/BiLLa.svg)
  * 简介：该项目开源了推理能力增强的中英双语LLaMA模型。模型的主要特性有：较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；全量参数更新，追求更好的生成效果。
* Moss：
  * 地址：https://github.com/OpenLMLab/MOSS
    ![](https://img.shields.io/github/stars/OpenLMLab/MOSS.svg)
  * 简介：支持中英双语和多种插件的开源对话语言模型，MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。
* Luotuo-Chinese-LLM：
  * 地址：https://github.com/LC1332/Luotuo-Chinese-LLM
    ![](https://img.shields.io/github/stars/LC1332/Luotuo-Chinese-LLM.svg)
  * 简介：囊括了一系列中文大语言模型开源项目，包含了一系列基于已有开源模型（ChatGLM, MOSS, LLaMA）进行二次微调的语言模型，指令微调数据集等。
* Linly：
  * 地址：https://github.com/CVI-SZU/Linly
    ![](https://img.shields.io/github/stars/CVI-SZU/Linly.svg)
  * 简介：提供中文对话模型 Linly-ChatFlow 、中文基础模型 Linly-Chinese-LLaMA 及其训练数据。 中文基础模型以 LLaMA 为底座，利用中文和中英平行增量预训练。项目汇总了目前公开的多语言指令数据，对中文模型进行了大规模指令跟随训练，实现了 Linly-ChatFlow 对话模型。
* Firefly：
  * 地址：https://github.com/yangjianxin1/Firefly
    ![](https://img.shields.io/github/stars/yangjianxin1/Firefly.svg)
  * 简介：Firefly(流萤) 是一个开源的中文大语言模型项目，开源包括数据、微调代码、多个基于Bloom、baichuan等微调好的模型等；支持全量参数指令微调、QLoRA低成本高效指令微调、LoRA指令微调；支持绝大部分主流的开源大模型，如百川baichuan、Ziya、Bloom、LLaMA等。持lora与base model进行权重合并，推理更便捷。
* ChatYuan
  * 地址：https://github.com/clue-ai/ChatYuan
    ![](https://img.shields.io/github/stars/clue-ai/ChatYuan.svg)
  * 简介：元语智能发布的一系列支持中英双语的功能型对话语言大模型，在微调数据、人类反馈强化学习、思维链等方面进行了优化。
* ChatRWKV：
  * 地址：https://github.com/BlinkDL/ChatRWKV
    ![](https://img.shields.io/github/stars/BlinkDL/ChatRWKV.svg)
  * 简介：开源了一系列基于RWKV架构的Chat模型（包括英文和中文），发布了包括Raven，Novel-ChnEng，Novel-Ch与Novel-ChnEng-ChnPro等模型，可以直接闲聊及进行诗歌，小说等创作，包括7B和14B等规模的模型。
* CPM-Bee
  * 地址：https://github.com/OpenBMB/CPM-Bee
    ![](https://img.shields.io/github/stars/OpenBMB/CPM-Bee.svg)
  * 简介：一个完全开源、允许商用的百亿参数中英文基座模型。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。
* TigerBot
  * 地址：https://github.com/TigerResearch/TigerBot
    ![](https://img.shields.io/github/stars/TigerResearch/TigerBot.svg)
  * 简介：一个多语言多任务的大规模语言模型(LLM)，开源了包括模型：TigerBot-7B, TigerBot-7B-base，TigerBot-180B，基本训练和推理代码，100G预训练数据，涵盖金融、法律、百科的领域数据以及API等。
* Aquila
  * 地址：https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila
    ![](https://img.shields.io/github/stars/FlagAI-Open/FlagAI.svg)
  * 简介：由智源研究院发布，Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。
* Aquila2
  * 地址：https://github.com/FlagAI-Open/Aquila2
    ![](https://img.shields.io/github/stars/FlagAI-Open/Aquila2.svg)
  * 简介：由智源研究院发布，Aquila2 系列，包括基础语言模型 Aquila2-7B，Aquila2-34B 和 Aquila2-70B-Expr ，对话模型 AquilaChat2-7B ，AquilaChat2-34B 和 AquilaChat2-70B-Expr，长文本对话模型AquilaChat2-7B-16k 和 AquilaChat2-34B-16。
* Anima
  * 地址：https://github.com/lyogavin/Anima
    ![](https://img.shields.io/github/stars/lyogavin/Anima.svg)
  * 简介：由艾写科技开发的一个开源的基于QLoRA的33B中文大语言模型，该模型基于QLoRA的Guanaco 33B模型使用Chinese-Vicuna项目开放的训练数据集guanaco_belle_merge_v1.0进行finetune训练了10000个step，基于Elo rating tournament评估效果较好。
* KnowLM
  * 地址：https://github.com/zjunlp/KnowLM
    ![](https://img.shields.io/github/stars/zjunlp/KnowLM.svg)
  * 简介：KnowLM项目旨在发布开源大模型框架及相应模型权重以助力减轻知识谬误问题，包括大模型的知识难更新及存在潜在的错误和偏见等。该项目一期发布了基于Llama的抽取大模型智析，使用中英文语料对LLaMA（13B）进行进一步全量预训练，并基于知识图谱转换指令技术对知识抽取任务进行优化。
* BayLing
  * 地址：https://github.com/ictnlp/BayLing
    ![](https://img.shields.io/github/stars/ictnlp/BayLing.svg)
  * 简介：一个具有增强的跨语言对齐的通用大模型，由中国科学院计算技术研究所自然语言处理团队开发。百聆（BayLing）以LLaMA为基座模型，探索了以交互式翻译任务为核心进行指令微调的方法，旨在同时完成语言间对齐以及与人类意图对齐，将LLaMA的生成能力和指令跟随能力从英语迁移到其他语言（中文）。在多语言翻译、交互翻译、通用任务、标准化考试的测评中，百聆在中文/英语中均展现出更好的表现。百聆提供了在线的内测版demo，以供大家体验。
* YuLan-Chat
  * 地址：https://github.com/RUC-GSAI/YuLan-Chat
    ![](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Chat.svg)
  * 简介：YuLan-Chat是中国人民大学GSAI研究人员开发的基于聊天的大语言模型。它是在LLaMA的基础上微调开发的，具有高质量的英文和中文指令。 YuLan-Chat可以与用户聊天，很好地遵循英文或中文指令，并且可以在量化后部署在GPU（A800-80G或RTX3090）上。
* PolyLM
  * 地址：https://github.com/DAMO-NLP-MT/PolyLM
    ![](https://img.shields.io/github/stars/DAMO-NLP-MT/PolyLM.svg)
  * 简介：一个在6400亿个词的数据上从头训练的多语言语言模型，包括两种模型大小(1.7B和13B)。PolyLM覆盖中、英、俄、西、法、葡、德、意、荷、波、阿、土、希伯来、日、韩、泰、越、印尼等语种，特别是对亚洲语种更友好。
* huozi
  * 地址：https://github.com/HIT-SCIR/huozi
    ![](https://img.shields.io/github/stars/HIT-SCIR/huozi.svg)
  * 简介：由哈工大自然语言处理研究所多位老师和学生参与开发的一个开源可商用的大规模预训练语言模型。 该模型基于 Bloom 结构的70 亿参数模型，支持中英双语，上下文窗口长度为 2048，同时还开源了基于RLHF训练的模型以及全人工标注的16.9K中文偏好数据集。
* YaYi
  * 地址：https://github.com/wenge-research/YaYi
    ![](https://img.shields.io/github/stars/wenge-research/YaYi.svg)
  * 简介：雅意大模型在百万级人工构造的高质量领域数据上进行指令微调得到，训练数据覆盖媒体宣传、舆情分析、公共安全、金融风控、城市治理等五大领域，上百种自然语言指令任务。雅意大模型从预训练初始化权重到领域模型的迭代过程中，我们逐步增强了它的中文基础能力和领域分析能力，并增加了多轮对话和部分插件能力。同时，经过数百名用户内测过程中持续不断的人工反馈优化，进一步提升了模型性能和安全性。已开源基于 LLaMA 2 的中文优化模型版本，探索适用于中文多领域任务的最新实践。
* YAYI2
  * 地址：https://github.com/wenge-research/YAYI2
    ![](https://img.shields.io/github/stars/wenge-research/YAYI2.svg)
  * 简介：YAYI 2 是中科闻歌研发的新一代开源大语言模型，包括 Base 和 Chat 版本，参数规模为 30B。YAYI2-30B 是基于 Transformer 的大语言模型，采用了超过 2 万亿 Tokens 的高质量、多语言语料进行预训练。针对通用和特定领域的应用场景，我们采用了百万级指令进行微调，同时借助人类反馈强化学习方法，以更好地使模型与人类价值观对齐。本次开源的模型为 YAYI2-30B Base 模型。
* Yuan-2.0
  * 地址：https://github.com/IEIT-Yuan/Yuan-2.0
    ![](https://img.shields.io/github/stars/IEIT-Yuan/Yuan-2.0.svg)
  * 简介：该项目开源了由浪潮信息发布的新一代基础语言大模型，具体开源了全部的3个模型源2.0-102B，源2.0-51B和源2.0-2B。并且提供了预训练，微调，推理服务的相关脚本。源2.0是在源1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同方面具备更强的理解能力。
* Chinese-Mixtral-8x7B
  * 地址：https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B
    ![](https://img.shields.io/github/stars/HIT-SCIR/Chinese-Mixtral-8x7B)
  * 简介：该项目基于Mixtral-8x7B稀疏混合专家模型进行了中文扩词表增量预训练，开源了Chinese-Mixtral-8x7B扩词表模型以及训练代码。该模型的的中文编解码效率较原模型显著提高。同时通过在大规模开源语料上进行的增量预训练，该模型具备了强大的中文生成和理解能力。
* BlueLM
  * 地址：https://github.com/vivo-ai-lab/BlueLM
    ![](https://img.shields.io/github/stars/vivo-ai-lab/BlueLM.svg)
  * 簡介：BlueLM 是由 vivo AI 全球研究院自主研发的大规模预训练语言模型，本次发布包含 7B 基础 (base) 模型和 7B 对话 (chat) 模型，同时我们开源了支持 32K 的长文本基础 (base) 模型和对话 (chat) 模型。
* TuringMM
  * 地址：https://github.com/lightyear-turing/TuringMM-34B-Chat
    ![](https://img.shields.io/github/stars/lightyear-turing/TuringMM-34B-Chat.svg)
  * 簡介：TuringMM-34B-Chat是一款开源的中英文Chat模型，由北京光年无限科技有限公司基于Yi-34B开源模型、基于14w的精标教育数据进行sft微调以及15W对齐数据进行DPO偏好学习得到的一个微调模型。
* Orion
  * 地址：https://github.com/OrionStarAI/Orion
    ![](https://img.shields.io/github/stars/OrionStarAI/Orion.svg)
  * 簡介：Orion-14B-Base是一个具有140亿参数的多语种大模型，该模型在一个包含2.5万亿token的多样化数据集上进行了训练，涵盖了中文、英语、日语、韩语等多种语言。
* OrionStar-Yi-34B-Chat
  * 地址：https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat
    ![](https://img.shields.io/github/stars/OrionStarAI/OrionStar-Yi-34B-Chat.svg)
  * 簡介：OrionStar-Yi-34B-Chat 是猎户星空基于零一万物开源的Yi-34B模型，使用 15W+ 的高质量语料训练而来微调大模型，旨在为大模型社区用户提供卓越的交互体验。
* MiniCPM
  * 地址：https://github.com/OpenBMB/MiniCPM
    ![](https://img.shields.io/github/stars/OpenBMB/MiniCPM.svg)
  * 简介：MiniCPM 是面壁智能与清华大学自然语言处理实验室共同开源的系列端侧大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量, 总计2.7B参数量。
* Mengzi3
  * 地址：https://github.com/Langboat/Mengzi3
    ![](https://img.shields.io/github/stars/Langboat/Mengzi3.svg)
  * 简介：Mengzi3 8B/13B模型基于Llama架构，语料精选自网页、百科、社交、媒体、新闻，以及高质量的开源数据集。通过在万亿tokens上进行多语言语料的继续训练，模型的中文能力突出并且兼顾多语言能力。

#### 1.2 多模态LLM模型

* VisualGLM-6B
  
  * 地址：https://github.com/THUDM/VisualGLM-6B
    ![](https://img.shields.io/github/stars/THUDM/VisualGLM-6B.svg)
  * 简介：一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。依靠来自于 CogView 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练。

* CogVLM
  
  * 地址：https://github.com/THUDM/CogVLM
    ![](https://img.shields.io/github/stars/THUDM/VisualGLM-6B.svg)
  * 简介：一个强大的开源视觉语言模型（VLM）。CogVLM-17B 拥有 100 亿视觉参数和 70 亿语言参数。 CogVLM-17B 在 10 个经典跨模态基准测试上取得了 SOTA 性能。CogVLM 能够准确地描述图像，几乎不会出现幻觉。

* Visual-Chinese-LLaMA-Alpaca
  
  * 地址：https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca
    ![](https://img.shields.io/github/stars/airaria/Visual-Chinese-LLaMA-Alpaca.svg)
  * 简介：基于中文LLaMA&Alpaca大模型项目开发的多模态中文大模型。VisualCLA在中文LLaMA/Alpaca模型上增加了图像编码等模块，使LLaMA模型可以接收视觉信息。在此基础上，使用了中文图文对数据进行了多模态预训练，对齐图像与文本表示，赋予其基本的多模态理解能力；并使用多模态指令数据集精调，增强其对多模态指令的理解、执行和对话能力，目前开源了VisualCLA-7B-v0.1。

* LLaSM
  
  * 地址：https://github.com/LinkSoul-AI/LLaSM
    ![](https://img.shields.io/github/stars/LinkSoul-AI/LLaSM.svg)
  * 简介：第一个支持中英文双语语音-文本多模态对话的开源可商用对话模型。便捷的语音输入将大幅改善以文本为输入的大模型的使用体验，同时避免了基于 ASR 解决方案的繁琐流程以及可能引入的错误。目前开源了LLaSM-Chinese-Llama-2-7B、LLaSM-Baichuan-7B等模型与数据集。

* VisCPM
  
  * 地址：https://github.com/OpenBMB/VisCPM
    ![](https://img.shields.io/github/stars/OpenBMB/VisCPM.svg)
  * 简介：一个开源的多模态大模型系列，支持中英双语的多模态对话能力（VisCPM-Chat模型）和文到图生成能力（VisCPM-Paint模型）。VisCPM基于百亿参数量语言大模型CPM-Bee（10B）训练，融合视觉编码器（Q-Former）和视觉解码器（Diffusion-UNet）以支持视觉信号的输入和输出。得益于CPM-Bee基座优秀的双语能力，VisCPM可以仅通过英文多模态数据预训练，泛化实现优秀的中文多模态能力。

* MiniCPM-V
  
  * 地址：https://github.com/OpenBMB/MiniCPM-V
    ![](https://img.shields.io/github/stars/OpenBMB/MiniCPM-V.svg)
  * 简介：面向图文理解的端侧多模态大模型系列。包括MiniCPM-V 2/2.6等系列，参数量包括2B，8B等，2B多模态综合性能超越 Yi-VL 34B、CogVLM-Chat 17B、Qwen-VL-Chat 10B 等更大参数规模的模型， 8B，单图、多图和视频理解性能超越了 GPT-4V。

* Qwen-VL
  
  * 地址：https://github.com/QwenLM/Qwen-VL
    ![](https://img.shields.io/github/stars/QwenLM/Qwen-VL.svg)
  * 简介：是阿里云研发的大规模视觉语言模型，可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。特点包括：强大的性能：在四大类多模态任务的标准英文测评中上均取得同等通用模型大小下最好效果；多语言对话模型：天然支持英文、中文等多语言对话，端到端支持图片里中英双语的长文本识别；多图交错对话：支持多图输入和比较，指定图片问答，多图文学创作等；首个支持中文开放域定位的通用模型：通过中文开放域语言表达进行检测框标注；细粒度识别和理解：相比于目前其它开源LVLM使用的224分辨率，Qwen-VL是首个开源的448分辨率的LVLM模型。更高分辨率可以提升细粒度的文字识别、文档问答和检测框标注。

* InternVL/1.5/2.0
  * 地址：https://github.com/OpenGVLab/InternVL
    ![](https://img.shields.io/github/stars/OpenGVLab/InternVL.svg)
  * 简介：开源多模态大模型，也是国内首个在MMMU（多学科问答）上突破60的模型。数学基准MathVista的测试中、书生·万象的得分为66.3%，显著高于其他闭源商业模型和开源模型。在通用图表基准ChartQA、文档类基准DocVQA、信息图表类基准InfographicVQA中以及通用视觉问答基准MMBench (v1.1)中，书生万象也取得了最先进（SOTA）的表现。

### 2. <a name='应用'></a>应用

#### 2.1 垂直领域微调

##### 医疗

[![](src/Medical.png)](src/Medical.png)

* DoctorGLM：
  
  * 地址：https://github.com/xionghonglin/DoctorGLM
    ![](https://img.shields.io/github/stars/xionghonglin/DoctorGLM.svg)
  * 简介：基于 ChatGLM-6B的中文问诊模型，通过中文医疗对话数据集进行微调，实现了包括lora、p-tuningv2等微调及部署

* BenTsao：
  
  * 地址：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese
    ![](https://img.shields.io/github/stars/SCIR-HI/Huatuo-Llama-Med-Chinese.svg)
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过医学知识图谱和GPT3.5 API构建了中文医学指令数据集，并在此基础上对LLaMA进行了指令微调，提高了LLaMA在医疗领域的问答效果。

* BianQue：
  
  * 地址：https://github.com/scutcyr/BianQue
    ![](https://img.shields.io/github/stars/scutcyr/BianQue.svg)
  * 简介：一个经过指令与多轮问询对话联合微调的医疗对话大模型，基于ClueAI/ChatYuan-large-v2作为底座，使用中文医疗问答指令与多轮问询对话混合数据集进行微调。

* HuatuoGPT：
  
  * 地址：https://github.com/FreedomIntelligence/HuatuoGPT
    ![](https://img.shields.io/github/stars/FreedomIntelligence/HuatuoGPT.svg)
  * 简介：开源了经过中文医学指令精调/指令微调(Instruct-tuning)的一个GPT-like模型

* Med-ChatGLM：
  
  * 地址：https://github.com/SCIR-HI/Med-ChatGLM
    ![](https://img.shields.io/github/stars/SCIR-HI/Med-ChatGLM.svg)
  * 简介：基于中文医学知识的ChatGLM模型微调，微调数据与BenTsao相同。

* QiZhenGPT：
  
  * 地址：https://github.com/CMKRG/QiZhenGPT
    ![](https://img.shields.io/github/stars/CMKRG/QiZhenGPT.svg)
  * 简介：该项目利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展。

* ChatMed：
  
  * 地址：https://github.com/michael-wzhu/ChatMed
    ![](https://img.shields.io/github/stars/michael-wzhu/ChatMed.svg)
  * 简介：该项目推出ChatMed系列中文医疗大规模语言模型，模型主干为LlaMA-7b并采用LoRA微调，具体包括ChatMed-Consult : 基于中文医疗在线问诊数据集ChatMed_Consult_Dataset的50w+在线问诊+ChatGPT回复作为训练集；ChatMed-TCM : 基于中医药指令数据集ChatMed_TCM_Dataset，以开源的中医药知识图谱为基础，采用以实体为中心的自指令方法(entity-centric self-instruct)，调用ChatGPT得到2.6w+的围绕中医药的指令数据训练得到。

* XrayGLM，首个会看胸部X光片的中文多模态医学大模型：
  
  * 地址：https://github.com/WangRongsheng/XrayGLM
    ![](https://img.shields.io/github/stars/WangRongsheng/XrayGLM.svg)
  * 简介：该项目为促进中文领域医学多模态大模型的研究发展，发布了XrayGLM数据集及模型，其在医学影像诊断和多轮交互对话上显示出了非凡的潜力。

* MeChat，中文心理健康支持对话大模型：
  
  * 地址：https://github.com/qiuhuachuan/smile
    ![](https://img.shields.io/github/stars/qiuhuachuan/smile.svg)
  * 简介：该项目开源的中文心理健康支持通用模型由 ChatGLM-6B LoRA 16-bit 指令微调得到。数据集通过调用gpt-3.5-turbo API扩展真实的心理互助 QA为多轮的心理健康支持多轮对话，提高了通用语言大模型在心理健康支持领域的表现，更加符合在长程多轮对话的应用场景。

* MedicalGPT
  
  * 地址：https://github.com/shibing624/MedicalGPT
    ![](https://img.shields.io/github/stars/shibing624/MedicalGPT.svg)
  * 简介：训练医疗大模型，实现包括二次预训练、有监督微调、奖励建模、强化学习训练。发布中文医疗LoRA模型shibing624/ziya-llama-13b-medical-lora，基于Ziya-LLaMA-13B-v1模型，SFT微调了一版医疗模型，医疗问答效果有提升，发布微调后的LoRA权重。

* Sunsimiao
  
  * 地址：https://github.com/thomas-yanxin/Sunsimiao
    ![](https://img.shields.io/github/stars/thomas-yanxin/Sunsimiao.svg)
  * 简介：Sunsimiao是一个开源的中文医疗大模型，该模型基于baichuan-7B和ChatGLM-6B底座模型在十万级高质量的中文医疗数据中微调而得。

* ShenNong-TCM-LLM
  
  * 地址：https://github.com/michael-wzhu/ShenNong-TCM-LLM
    ![](https://img.shields.io/github/stars/michael-wzhu/ShenNong-TCM-LLM.svg)
  * 简介：该项目开源了ShenNong中医药大规模语言模型，该模型以LlaMA为底座，采用LoRA (rank=16)微调得到。微调代码与ChatMed代码库相同。此外该项目还开源了中医药指令微调数据集。

* SoulChat
  
  * 地址：https://github.com/scutcyr/SoulChat
    ![](https://img.shields.io/github/stars/scutcyr/SoulChat.svg)
  * 简介：该项目开源了经过百万规模心理咨询领域中文长文本指令与多轮共情对话数据联合指令微调的心理健康大模型灵心（SoulChat），该模型以ChatGLM-6B作为初始化模型，进行了全量参数的指令微调。

* CareGPT
  
  * 地址：https://github.com/WangRongsheng/CareGPT
    ![](https://img.shields.io/github/stars/WangRongsheng/CareGPT.svg)
  * 简介：该项目开源了数十个公开可用的医疗微调数据集和开放可用的医疗大语言模型，包含LLM的训练、测评、部署等以促进医疗LLM快速发展。

* DISC-MedLLM
  
  * 地址：https://github.com/FudanDISC/DISC-MedLLM
    ![](https://img.shields.io/github/stars/FudanDISC/DISC-MedLLM.svg)
  * 简介：该项目是由复旦大学发布的针对医疗健康对话式场景而设计的医疗领域大模型与数据集，该模型由DISC-Med-SFT数据集基于Baichuan-13B-Base指令微调得到。

* Taiyi-LLM
  
  * 地址：https://github.com/DUTIR-BioNLP/Taiyi-LLM
    ![](https://img.shields.io/github/stars/DUTIR-BioNLP/Taiyi-LLM.svg)
  * 简介：该项目由大连理工大学信息检索研究室开发的中英双语医学大模型"太一"，收集整理了丰富的中英双语生物医学自然语言处理（BioNLP）训练语料，总共包含38个中文数据集，通过丰富的中英双语任务指令数据（超过100W条样本）进行大模型（Qwen-7B-base）指令微调，使模型具备了出色的中英双语生物医学智能问答、医患对话、报告生成、信息抽取、机器翻译、标题生成、文本分类等多种BioNLP能力。

* WiNGPT
  
  * 地址：https://github.com/winninghealth/WiNGPT2
    ![](https://img.shields.io/github/stars/winninghealth/WiNGPT2.svg)
  * 简介：WiNGPT是一个基于GPT的医疗垂直领域大模型，基于Qwen-7b1作为基础预训练模型，在此技术上进行了继续预训练，指令微调等，该项目具体开源了WiNGPT2-7B-Base与WiNGPT2-7B-Chat模型。

* ChiMed-GPT
  
  * 地址：https://github.com/synlp/ChiMed-GPT
    ![](https://img.shields.io/github/stars/synlp/ChiMed-GPT.svg)
  * 简介：ChiMed-GPT是一个开源中文医学大语言模型，通过在中文医学数据上持续训练 Ziya-v2 构建而成，其中涵盖了预训练、有监督微调 (SFT) 和来自人类反馈的强化学习 (RLHF) 等训练过程。

* MindChat
  
  * 地址：https://github.com/X-D-Lab/MindChat
    ![](https://img.shields.io/github/stars/X-D-Lab/MindChat.svg)
  * 简介：心理大模型——漫谈(MindChat)期望从心理咨询、心理评估、心理诊断、心理治疗四个维度帮助人们纾解心理压力与解决心理困惑，为用户提供隐私、温暖、安全、及时、方便的对话环境，从而帮助用户克服各种困难和挑战，实现自我成长和发展。MindChat是一个基于Qwen作为基础预训练模型，并在此基础上进行指令微调得到的心理垂域大模型。

##### 法律

[![](src/Legal.png)](src/Legal.png)

* 獬豸(LawGPT_zh): 中文法律对话语言模型
  
  * 地址：https://github.com/LiuHC0428/LAW-GPT
    ![](https://img.shields.io/github/stars/LiuHC0428/LAW-GPT.svg)
  * 简介: 本项目开源的中文法律通用模型由ChatGLM-6B LoRA 16-bit指令微调得到。数据集包括现有的法律问答数据集和基于法条和真实案例指导的self-Instruct构建的高质量法律文本问答，提高了通用语言大模型在法律领域的表现，提高了模型回答的可靠性和专业程度。

* LaWGPT：基于中文法律知识的大语言模型
  
  * 地址：https://github.com/pengxiao-song/LaWGPT
    ![](https://img.shields.io/github/stars/pengxiao-song/LaWGPT.svg)
  * 简介：该系列模型在通用中文基座模型（如 Chinese-LLaMA、ChatGLM 等）的基础上扩充法律领域专有词表、大规模中文法律语料预训练，增强了大模型在法律领域的基础语义理解能力。在此基础上，构造法律领域对话问答数据集、中国司法考试数据集进行指令精调，提升了模型对法律内容的理解和执行能力。

* LexiLaw：中文法律大模型
  
  * 地址：https://github.com/CSHaitao/LexiLaw
    ![](https://img.shields.io/github/stars/CSHaitao/LexiLaw.svg)
  * 简介：LexiLaw 是一个基于 ChatGLM-6B微调的中文法律大模型，通过在法律领域的数据集上进行微调。该模型旨在为法律从业者、学生和普通用户提供准确、可靠的法律咨询服务，包括具体法律问题的咨询，还是对法律条款、案例解析、法规解读等方面的查询。

* Lawyer LLaMA：中文法律LLaMA
  
  * 地址：https://github.com/AndrewZhe/lawyer-llama
    ![](https://img.shields.io/github/stars/AndrewZhe/lawyer-llama.svg)
  * 简介：开源了一系列法律领域的指令微调数据和基于LLaMA训练的中文法律大模型的参数。Lawyer LLaMA 首先在大规模法律语料上进行了continual pretraining。在此基础上，借助ChatGPT收集了一批对中国国家统一法律职业资格考试客观题（以下简称法考）的分析和对法律咨询的回答，利用收集到的数据对模型进行指令微调，让模型习得将法律知识应用到具体场景中的能力。

* 韩非(HanFei)
  
  * 地址: https://github.com/siat-nlp/HanFei
    ![](https://img.shields.io/github/stars/siat-nlp/HanFei.svg)
  * 简介: HanFei-1.0(韩非)是国内首个全参数训练的法律大模型，参数量7b，主要功能包括：法律问答、多轮对话、撰写文章、检索等。

* ChatLaw-法律大模型
  
  * 地址：https://github.com/PKU-YuanGroup/ChatLaw
    ![](https://img.shields.io/github/stars/PKU-YuanGroup/ChatLaw.svg)
  * 简介：由北大开源的一系列法律领域的大模型，包括ChatLaw-13B（基于姜子牙Ziya-LLaMA-13B-v1训练而来），ChatLaw-33B（基于Anima-33B训练而来，逻辑推理能力大幅提升），ChatLaw-Text2Vec，使用93w条判决案例做成的数据集基于BERT训练了一个相似度匹配模型，可将用户提问信息和对应的法条相匹配。

* lychee_law-律知
  
  * 地址：https://github.com/davidpig/lychee_law
    ![](https://img.shields.io/github/stars/davidpig/lychee_law.svg)
  * 简介：该项目由德国萨尔大学团队和中国南京大学团队合作开发，开源一系列中文司法领域大模型，如Law-GLM-10B: 基于 GLM-10B 模型, 在 30GB 中文法律数据上进行指令微调得到的。

* 智海-录问(wisdomInterrogatory)
  
  * 地址：https://github.com/zhihaiLLM/wisdomInterrogatory
    ![](https://img.shields.io/github/stars/zhihaiLLM/wisdomInterrogatory.svg)
  * 简介：该项目由浙江大学、阿里巴巴达摩院以及华院计算三家单位共同设计研发的法律大模型，基于baichuan-7b进行了法律领域数据的二次预训练与指令微调，并设计了知识增强的推理流程。

* 夫子•明察司法大模型
  
  * 地址：https://github.com/irlab-sdu/fuzi.mingcha
    ![](https://img.shields.io/github/stars/irlab-sdu/fuzi.mingcha.svg)
  * 简介：该项目由是由山东大学、浪潮云、中国政法大学联合研发，以 ChatGLM 为大模型底座，基于海量中文无监督司法语料（包括各类判决文书、法律法规等）与有监督司法微调数据（包括法律问答、类案检索）训练的中文司法大模型。该模型支持法条检索、案例分析、三段论推理判决以及司法对话等功能。

* DISC-LawLLM
  
  * 地址：https://github.com/FudanDISC/DISC-LawLLM
    ![](https://img.shields.io/github/stars/FudanDISC/DISC-LawLLM.svg)
  * 简介：该项目由由复旦大学数据智能与社会计算实验室 (Fudan-DISC) 开发并开源的法律领域大模型，包括数据集，基于 Baichuan-13B-Base 进行微调的模型，且增加了检索增强模块。

##### 金融

[![](src/Financial.png)](src/Financial.png)

* Cornucopia（聚宝盆）：基于中文金融知识的LLaMA微调模型
  
  * 地址：https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese
    ![](https://img.shields.io/github/stars/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese.svg)
  * 简介：开源了经过中文金融知识指令精调/指令微调(Instruct-tuning) 的LLaMA-7B模型。通过中文金融公开数据+爬取的金融数据构建指令数据集，并在此基础上对LLaMA进行了指令微调，提高了 LLaMA 在金融领域的问答效果。基于相同的数据，后期还会利用GPT3.5 API构建高质量的数据集，另在中文知识图谱-金融上进一步扩充高质量的指令数据集。

* BBT-FinCUGE-Applications
  
  * 地址：https://github.com/ssymmetry/BBT-FinCUGE-Applications
    ![](https://img.shields.io/github/stars/ssymmetry/BBT-FinCUGE-Applications.svg)
  * 简介：开源了中文金融领域开源语料库BBT-FinCorpus，中文金融领域知识增强型预训练语言模型BBT-FinT5及中文金融领域自然语言处理评测基准CFLEB。

* XuanYuan（轩辕）：首个千亿级中文金融对话模型
  
  * 地址：https://github.com/Duxiaoman-DI/XuanYuan
    ![](https://img.shields.io/github/stars/Duxiaoman-DI/XuanYuan.svg)
  * 简介：轩辕是国内首个开源的千亿级中文对话大模型，同时也是首个针对中文金融领域优化的千亿级开源对话大模型。轩辕在BLOOM-176B的基础上针对中文通用领域和金融领域进行了针对性的预训练与微调，它不仅可以应对通用领域的问题，也可以解答与金融相关的各类问题，为用户提供准确、全面的金融信息和建议。

* FinGPT
  
  * 地址：https://github.com/AI4Finance-Foundation/FinGPT
    ![](https://img.shields.io/github/stars/AI4Finance-Foundation/FinGPT.svg)
  * 简介：该项目开源了多个金融大模型，包括ChatGLM-6B/ChatGLM2-6B+LoRA和LLaMA-7B+LoRA的金融大模型，收集了包括金融新闻、社交媒体、财报等中英文训练数据。

* DISC-FinLLM
  
  * 地址：https://github.com/FudanDISC/DISC-FinLLM
    ![](https://img.shields.io/github/stars/FudanDISC/DISC-FinLLM.svg)
  * 简介：该项目由复旦大学数据智能与社会计算实验室 (Fudan-DISC) 开发并开源，项目中开源的资源包括：DISC-FinLLM-SFT训练数据样本，DISC-FinLLM模型参数（基于Baichuan-13B-Chat训练），DISC-Fin-Eval-Benchmark等。

* Tongyi-Finance
  
  * 地址：https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B
  * 简介：该模型是针对对金融行业推出的大语言模型，基于通义千问基础模型进行行业语料增量学习，强化金融领域知识和场景应用能力，覆盖金融知识问答、文本分类、信息抽取、文本创作、阅读理解、逻辑推理、多模态、Coding等能力象限。具有以下特点：行业语料增量学习：使用200B高质量金融行业语料进行增量学习，并进行金融行业词表扩展，覆盖丰富的数据类型，支持更大上下文（16k）输入和完整的语义表达。行业能力强化：自研SFT质量&多样性分析工具，筛选高质量SFT数据，解决大语言模型的alignment问题。行业后链路优化：借助multi-agent框架，实现知识库增强和工具API调用。

##### 教育

* 桃李（Taoli）：
  
  * 地址：https://github.com/blcuicall/taoli
    ![](https://img.shields.io/github/stars/blcuicall/taoli.svg)
  * 简介：一个在国际中文教育领域数据上进行了额外训练的模型。项目基于目前国际中文教育领域流通的500余册国际中文教育教材与教辅书、汉语水平考试试题以及汉语学习者词典等，构建了国际中文教育资源库，构造了共计 88000 条的高质量国际中文教育问答数据集，并利用收集到的数据对模型进行指令微调，让模型习得将知识应用到具体场景中的能力。

* EduChat：
  
  * 地址：https://github.com/icalk-nlp/EduChat
    ![](https://img.shields.io/github/stars/icalk-nlp/EduChat.svg)
  * 简介：该项目华东师范大学计算机科学与技术学院的EduNLP团队研发，主要研究以预训练大模型为基底的教育对话大模型相关技术，融合多样化的教育垂直领域数据，辅以指令微调、价值观对齐等方法，提供教育场景下自动出题、作业批改、情感支持、课程辅导、高考咨询等丰富功能，服务于广大老师、学生和家长群体，助力实现因材施教、公平公正、富有温度的智能教育。

* chatglm-maths：
  
  * 地址：https://github.com/yongzhuo/chatglm-maths
    ![](https://img.shields.io/github/stars/yongzhuo/chatglm-maths.svg)
  * 简介：基于chatglm-6b微调/LORA/PPO/推理的数学题解题大模型, 样本为自动生成的整数/小数加减乘除运算, 可gpu/cpu部署，开源了训练数据集等。

* MathGLM：
  
  * 地址：https://github.com/THUDM/MathGLM
    ![](https://img.shields.io/github/stars/THUDM/MathGLM.svg)
  * 简介：该项目由THUDM研发，开源了多个能进行20亿参数可以进行准确多位算术运算的语言模型，同时开源了可用于算术运算微调的数据集。

* QiaoBan：
  
  * 地址：https://github.com/HIT-SCIR-SC/QiaoBan
    ![](https://img.shields.io/github/stars/HIT-SCIR-SC/QiaoBan.svg)
  * 简介：该项目旨在构建一个面向儿童情感陪伴的大模型，这个仓库包含：用于指令微调的对话数据/data，巧板的训练代码，训练配置文件，使用巧板进行对话的示例代码（TODO，checkpoint将发布至huggingface）。

##### 科技

* 天文大语言模型StarGLM：
  
  * 地址：https://github.com/Yu-Yang-Li/StarGLM
    ![](https://img.shields.io/github/stars/Yu-Yang-Li/StarGLM.svg)
  * 简介：基于ChatGLM训练了天文大语言模型，以期缓解大语言模型在部分天文通用知识和前沿变星领域的幻觉现象，为接下来可处理天文多模态任务、部署于望远镜阵列的观测Agent——司天大脑（数据智能处理）打下基础。

* TransGPT·致远：
  
  * 地址：https://github.com/DUOMO/TransGPT
    ![](https://img.shields.io/github/stars/DUOMO/TransGPT.svg)
  * 简介：开源交通大模型，主要致力于在真实交通行业中发挥实际价值。它能够实现交通情况预测、智能咨询助手、公共交通服务、交通规划设计、交通安全教育、协助管理、交通事故报告和分析、自动驾驶辅助系统等功能。

* Mozi：
  
  * 地址：https://github.com/gmftbyGMFTBY/science-llm
    ![](https://img.shields.io/github/stars/gmftbyGMFTBY/science-llm.svg)
  * 简介：该项目开源了基于LLaMA和Baichuan的科技论文大模型，可以用于科技文献的问答和情感支持。

##### 电商

* EcomGPT
  * 地址：https://github.com/Alibaba-NLP/EcomGPT
    ![](https://img.shields.io/github/stars/Alibaba-NLP/EcomGPT.svg)
  * 简介：一个由阿里发布的面向电商领域的语言模型，该模型基于BLOOMZ在电商指令微调数据集上微调得到，人工评估在12个电商评测数据集上超过ChatGPT。

##### 网络安全

* SecGPT
  * 地址：https://github.com/Clouditera/secgpt
    ![](https://img.shields.io/github/stars/Clouditera/secgpt.svg)
  * 简介：开项目开源了网络安全大模型，该模型基于Baichuan-13B采用Lora做预训练和SFT训练，此外该项目还开源了相关预训练和指令微调数据集等资源。

##### 农业

* 后稷（AgriMa）：
  * 地址：https://github.com/zhiweihu1103/AgriMa
    ![](https://img.shields.io/github/stars/zhiweihu1103/AgriMa.svg)
  * 简介：首个中文开源农业大模型是由山西大学、山西农业大学与The Fin AI联合研发，以Baichuan为底座，基于海量有监督农业领域相关数据微调，具备广泛的农业知识和智能分析能力，该模型旨在为农业领域提供全面而高效的信息处理和决策支持。
* 稷丰（AgriAgent）：
  * 地址：https://github.com/zhiweihu1103/AgriAgent
  ![](https://img.shields.io/github/stars/zhiweihu1103/AgriAgent.svg)
  * 简介：首个开源中文农业多模态大模型是由山西农业大学研发，以[MiniCPM-Llama3-V 2.5](https://github.com/OpenBMB/MiniCPM-V)为底座，能够从图像、文本、气象数据等多源信息中提取有用信息，为农业生产提供全面、精准的智能化解决方案。我们致力于将稷丰应用于作物健康监测、病虫害识别、土壤肥力分析、农田管理优化等多个方面，帮助农民提升生产效率，减少资源浪费，促进农业的可持续发展。

#### 2.2 LangChain应用

* langchain-ChatGLM：
  
  * 地址：https://github.com/imClumsyPanda/langchain-ChatGLM
    ![](https://img.shields.io/github/stars/imClumsyPanda/langchain-ChatGLM.svg)
  * 简介：基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。建立了全流程可使用开源模型实现的本地知识库问答应用。现已支持使用 ChatGLM-6B 等大语言模型直接接入，或通过 fastchat api 形式接入 Vicuna, Alpaca, LLaMA, Koala, RWKV 等模型。

* LangChain-ChatGLM-Webui：
  
  * 地址：https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui
    ![](https://img.shields.io/github/stars/thomas-yanxin/LangChain-ChatGLM-Webui.svg)
  * 简介：利用LangChain和ChatGLM-6B系列模型制作的Webui, 提供基于本地知识的大模型应用。目前支持上传 txt、docx、md、pdf等文本格式文件, 提供包括ChatGLM-6B系列、Belle系列等模型文件以及GanymedeNil/text2vec-large-chinese、nghuyong/ernie-3.0-base-zh、nghuyong/ernie-3.0-nano-zh等Embedding模型。

* Langchain-ChatGLM-and-TigerBot：
  
  * 地址：https://github.com/wordweb/langchain-ChatGLM-and-TigerBot
    ![](https://img.shields.io/github/stars/wordweb/langchain-ChatGLM-and-TigerBot.svg)
  * 简介：该项目在langchain-ChatGLM的基础上补充了加载TigerBot模型的基于本地知识库的问答应用。

* Chinese-LangChain：
  
  * 地址：https://github.com/yanqiangmiffy/Chinese-LangChain
    ![](https://img.shields.io/github/stars/yanqiangmiffy/Chinese-LangChain.svg)
  * 简介：基于ChatGLM-6b+langchain实现本地化知识库检索与智能答案生成（包括互联网检索结果接入）

* Lagent：
  
  * 地址：https://github.com/InternLM/lagent
    ![](https://img.shields.io/github/stars/InternLM/lagent.svg)
  * 简介：Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体。具体实现了多种类型的智能体，如经典的 ReAct，AutoGPT 和 ReWoo 等智能体。框架简单易拓展. 只需要不到20行代码你就能够创造出一个你自己的智能体（agent）。同时支持了 Python 解释器、API 调用和搜索三类常用典型工具。灵活支持多个大语言模型. 提供了多种大语言模型支持包括 InternLM、Llama-2 等开源模型和 GPT-4/3.5 等基于 API 的闭源模型。

* DemoGPT：
  
  * 地址：https://github.com/melih-unsal/DemoGPT
    ![](https://img.shields.io/github/stars/melih-unsal/DemoGPT.svg)
  * 简介：⚡ DemoGPT 使您只需使用提示即可创建快速演示。 ⚡

* ChatDev：
  
  * 地址：https://github.com/OpenBMB/ChatDev
    ![](https://img.shields.io/github/stars/OpenBMB/ChatDev.svg)
  * 简介：ChatDev是一家虚拟软件公司，通过担任不同角色的各种智能代理进行运营，包括首席执行官、首席技术官、程序员、测试员等。 这些代理形成了一个多代理组织结构，并因“通过编程彻底改变数字世界”的使命而团结在一起。 ChatDev中的代理通过参加专门的功能研讨会进行协作，包括设计、编码、测试和记录等任务。

#### 2.3 其他应用

* wenda：
  
  * 地址：https://github.com/wenda-LLM/wenda
    ![](https://img.shields.io/github/stars/wenda-LLM/wenda.svg)
  * 简介：一个LLM调用平台。为小模型外挂知识库查找和设计自动执行动作，实现不亚于于大模型的生成能力。

* JittorLLMs：
  
  * 地址：https://github.com/Jittor/JittorLLMs
    ![](https://img.shields.io/github/stars/Jittor/JittorLLMs.svg)
  * 简介：计图大模型推理库：笔记本没有显卡也能跑大模型，具有成本低，支持广，可移植，速度快等优势。

* LMFlow:
  
  * 地址：https://github.com/OptimalScale/LMFlow
    ![](https://img.shields.io/github/stars/OptimalScale/LMFlow.svg)
  * 简介：LMFlow是香港科技大学LMFlow团队开发的大模型微调工具箱。LMFlow工具箱具有可扩展性强、高效、方便的特性。LMFlow仅使用180K条数据微调，即可得到在Huggingface榜单第一名的Robin模型。LMFlow支持用户快速训练个性化模型，仅需单张3090和5个小时即可微调70亿参数定制化模型。

* fastllm：
  
  * 地址：https://github.com/ztxz16/fastllm
    ![](https://img.shields.io/github/stars/ztxz16/fastllm.svg)
  * 简介：纯c++的全平台llm加速库，chatglm-6B级模型单卡可达10000+token / s，支持moss, chatglm, baichuan模型，手机端流畅运行。

* WebCPM
  
  * 地址：https://github.com/thunlp/WebCPM
    ![](https://img.shields.io/github/stars/thunlp/WebCPM.svg)
  * 简介：一个支持可交互网页搜索的中文大模型。 

* GPT Academic：
  
  * 地址：https://github.com/binary-husky/gpt_academic
    ![](https://img.shields.io/github/stars/binary-husky/gpt_academic.svg)
  * 简介：为GPT/GLM提供图形交互界面，特别优化论文阅读润色体验，支持并行问询多种LLM模型，支持清华chatglm等本地模型。兼容复旦MOSS, llama, rwkv, 盘古等。

* ChatALL：
  
  * 地址：https://github.com/sunner/ChatALL
    ![](https://img.shields.io/github/stars/sunner/ChatALL.svg)
  * 简介：ChatALL（中文名：齐叨）可以把一条指令同时发给多个 AI，可以帮助用户发现最好的回答。

* CreativeChatGLM：
  
  * 地址：https://github.com/ypwhs/CreativeChatGLM
    ![](https://img.shields.io/github/stars/ypwhs/CreativeChatGLM.svg)
  * 简介：可以使用修订和续写的功能来生成创意内容，可以使用“续写”按钮帮 ChatGLM 想一个开头，并让它继续生成更多的内容，你可以使用“修订”按钮修改最后一句 ChatGLM 的回复。

* docker-llama2-chat：
  
  * 地址：https://github.com/soulteary/docker-llama2-chat
    ![](https://img.shields.io/github/stars/soulteary/docker-llama2-chat.svg)
  * 简介：开源了一个只需要三步就可以上手LLaMA2的快速部署方案。

* ChatGLM2-Voice-Cloning：
  
  * 地址：https://github.com/KevinWang676/ChatGLM2-Voice-Cloning
    ![](https://img.shields.io/github/stars/KevinWang676/ChatGLM2-Voice-Cloning.svg)
  * 简介：实现了一个可以和喜欢的角色沉浸式对话的应用，主要采用ChatGLM2+声音克隆+视频对话的技术。

* Flappy
  
  * 地址：https://github.com/pleisto/flappy
    ![](https://img.shields.io/github/stars/pleisto/flappy.svg)
  * 简介：一个产品级面向所有程序员的LLM SDK，
 
* LazyLLM
  
  * 地址：[https://github.com/LazyAGI/LazyLLM](https://github.com/LazyAGI/LazyLLM)
    ![](https://img.shields.io/github/stars/LazyAGI/LazyLLM.svg)
  * 简介：LazyLLM是一款低代码构建多Agent大模型应用的开发工具，协助开发者用极低的成本构建复杂的AI应用，并可以持续的迭代优化效果。LazyLLM提供了更为灵活的应用功能定制方式，并实现了一套轻量级网管机制来支持一键部署多Agent应用，支持流式输出，兼容多个Iaas平台，且支持对应用中的模型进行持续微调。
 
* MemFree
  
  * 地址：[https://github.com/memfreeme/memfree](https://github.com/memfreeme/memfree)
    ![](https://img.shields.io/github/stars/memfreeme/memfree.svg)
  * 简介：MemFree 是一个开源的 Hybrid AI 搜索引擎，可以同时对您的个人知识库（如书签、笔记、文档等）和互联网进行搜索, 为你提供最佳答案。MemFree 支持自托管的极速无服务器向量数据库，支持自托管的极速Local Embedding and Rerank Service，支持一键部署。

### 3. <a name='数据集'></a>数据集

#### 预训练数据集

* MNBVC
  
  * 地址：https://github.com/esbatmop/MNBVC
    ![](https://img.shields.io/github/stars/esbatmop/MNBVC.svg)
  * 数据集说明：超大规模中文语料集，不但包括主流文化，也包括各个小众文化甚至火星文的数据。MNBVC数据集包括新闻、作文、小说、书籍、杂志、论文、台词、帖子、wiki、古诗、歌词、商品介绍、笑话、糗事、聊天记录等一切形式的纯文本中文数据。数据均来源于互联网收集，且在持续更新中。

* WuDaoCorporaText
  
  * 地址：https://data.baai.ac.cn/details/WuDaoCorporaText
  * 数据集说明：WuDaoCorpora是北京智源人工智能研究院（智源研究院）构建的大规模、高质量数据集，用于支撑大模型训练研究。目前由文本、对话、图文对、视频文本对四部分组成，分别致力于构建微型语言世界、提炼对话核心规律、打破图文模态壁垒、建立视频文字关联，为大模型训练提供坚实的数据支撑。

* CLUECorpus2020
  
  * 地址：https://github.com/CLUEbenchmark/CLUECorpus2020
    ![](https://img.shields.io/github/stars/CLUEbenchmark/CLUECorpus2020.svg)
  * 数据集说明：通过对Common Crawl的中文部分进行语料清洗，最终得到100GB的高质量中文预训练语料，可直接用于预训练、语言模型或语言生成任务以及专用于简体中文NLP任务的小词表。

* WanJuan-1.0
  
  * 地址：https://opendatalab.org.cn/WanJuan1.0
  * 数据集说明：书生·万卷1.0为书生·万卷多模态语料库的首个开源版本，包含文本数据集、图文数据集、视频数据集三部分，数据总量超过2TB。 目前，书生·万卷1.0已被应用于书生·多模态、书生·浦语的训练。通过对高质量语料的“消化”，书生系列模型在语义理解、知识问答、视觉理解、视觉问答等各类生成式任务表现出的优异性能。

* seq-monkey-data
  
  * 地址：https://github.com/mobvoi/seq-monkey-data
    
    ![](https://img.shields.io/github/stars/mobvoi/seq-monkey-data.svg)
  
  * 数据集说明：序列猴子是出门问问提供的超大规模语言模型，基于其通用的表示与推理能力，支持多轮交互，能够大幅度提高生产效率和数据处理能力，被广泛应用于问答系统、自然语言处理、机器翻译、文本摘要等领域。序列猴子数据集是用于训练序列猴子模型的数据集合，现选择部分数据集向公众开放。

#### SFT数据集

* RefGPT：基于RefGPT生成大量真实和定制的对话数据集
  
  * 地址：https://github.com/DA-southampton/RedGPT
    ![](https://img.shields.io/github/stars/DA-southampton/RedGPT.svg)
  * 数据集说明：包括RefGPT-Fact和RefGPT-Code两部分，其中RefGPT-Fact给出了5万中文的关于事实性知识的多轮对话，RefGPT-Code给出了3.9万中文编程相关的多轮对话数据。

* COIG
  
  * 地址：https://huggingface.co/datasets/BAAI/COIG
  * 数据集说明：维护了一套无害、有用且多样化的中文指令语料库，包括一个人工验证翻译的通用指令语料库、一个人工标注的考试指令语料库、一个人类价值对齐指令语料库、一个多轮反事实修正聊天语料库和一个 leetcode 指令语料库。

* generated_chat_0.4M：
  
  * 地址：https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M
  * 数据集说明：包含约40万条由BELLE项目生成的个性化角色对话数据，包含角色介绍。但此数据集是由ChatGPT产生的，未经过严格校验，题目或解题过程可能包含错误。

* alpaca_chinese_dataset：
  
  * 地址：https://github.com/hikariming/alpaca_chinese_dataset
    ![](https://img.shields.io/github/stars/hikariming/alpaca_chinese_dataset.svg)
  * 数据集说明：根据斯坦福开源的alpaca数据集进行中文翻译，并再制造一些对话数据

* Alpaca-CoT：
  
  * 地址：https://github.com/PhoebusSi/Alpaca-CoT
    ![](https://img.shields.io/github/stars/PhoebusSi/Alpaca-CoT.svg)
  * 数据集说明：统一了丰富的IFT数据（如CoT数据，目前仍不断扩充）、多种训练效率方法（如lora，p-tuning）以及多种LLMs，三个层面上的接口，打造方便研究人员上手的LLM-IFT研究平台。

* pCLUE：
  
  * 地址：https://github.com/CLUEbenchmark/pCLUE
    ![](https://img.shields.io/github/stars/CLUEbenchmark/pCLUE.svg)
  * 数据集说明：基于提示的大规模预训练数据集，用于多任务学习和零样本学习。包括120万训练数据，73个Prompt，9个任务。

* firefly-train-1.1M：
  
  * 地址：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
  * 数据集说明：23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万

* BELLE-data-1.5M：
  
  * 地址：https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M
    ![](https://img.shields.io/github/stars/LianjiaTech/BELLE.svg)
  * 数据集说明：通过self-instruct生成，使用了中文种子任务，以及openai的text-davinci-003接口,涉及175个种子任务

* Chinese Scientific Literature Dataset：
  
  * 地址：https://github.com/ydli-ai/csl
    ![](https://img.shields.io/github/stars/ydli-ai/csl.svg)
  * 数据集说明：中文科学文献数据集（CSL），包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）以及简单的prompt

* Chinese medical dialogue data：
  
  * 地址：https://github.com/Toyhom/Chinese-medical-dialogue-data
    ![](https://img.shields.io/github/stars/Toyhom/Chinese-medical-dialogue-data.svg)
  * 数据集说明：中文医疗对话数据集，包括：<Andriatria_男科> 94596个问答对 <IM_内科> 220606个问答对 <OAGD_妇产科> 183751个问答对 <Oncology_肿瘤科> 75553个问答对 <Pediatric_儿科> 101602个问答对 <Surgical_外科> 115991个问答对 总计 792099个问答对。

* Huatuo-26M：
  
  * 地址：https://github.com/FreedomIntelligence/Huatuo-26M
    ![](https://img.shields.io/github/stars/FreedomIntelligence/Huatuo-26M.svg)
  * 数据集说明：Huatuo-26M 是一个中文医疗问答数据集，此数据集包含了超过2600万个高质量的医疗问答对，涵盖了各种疾病、症状、治疗方式、药品信息等多个方面。Huatuo-26M 是研究人员、开发者和企业为了提高医疗领域的人工智能应用，如聊天机器人、智能诊断系统等需要的重要资源。

* Alpaca-GPT-4:
  
  * 地址：https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
    ![](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM.svg)
  * 数据集说明：Alpaca-GPT-4 是一个使用 self-instruct 技术，基于 175 条中文种子任务和 GPT-4 接口生成的 50K 的指令微调数据集。

* InstructionWild
  
  * 地址：https://github.com/XueFuzhao/InstructionWild
    ![](https://img.shields.io/github/stars/XueFuzhao/InstructionWild.svg)
  * 数据集说明：InstructionWild 是一个从网络上收集自然指令并过滤之后使用自然指令结合 ChatGPT 接口生成指令微调数据集的项目。主要的指令来源：Twitter、CookUp.AI、Github 和 Discard。

* ShareChat
  
  * 地址：https://paratranz.cn/projects/6725
  * 数据集说明：一个倡议大家一起翻译高质量 ShareGPT 数据的项目。
  * 项目介绍：清洗/构造/翻译中文的ChatGPT数据，推进国内AI的发展，人人可炼优质中文 Chat 模型。本数据集为ChatGPT约九万个对话数据，由ShareGPT API获得（英文68000，中文11000条，其他各国语言）。项目所有数据最终将以 CC0 协议并入 Multilingual Share GPT 语料库。

* Guanaco
  
  * 地址：https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
  * 数据集说明：一个使用 Self-Instruct 的主要包含中日英德的多语言指令微调数据集。

* chatgpt-corpus
  
  * 地址：https://github.com/PlexPt/chatgpt-corpus
    ![](https://img.shields.io/github/stars/PlexPt/chatgpt-corpus.svg)
  * 数据集说明：开源了由 ChatGPT3.5 生成的300万自问自答数据，包括多个领域，可用于用于训练大模型。

* SmileConv
  
  * 地址：https://github.com/qiuhuachuan/smile
    ![](https://img.shields.io/github/stars/qiuhuachuan/smile.svg)
  * 数据集说明：数据集通过ChatGPT改写真实的心理互助 QA为多轮的心理健康支持多轮对话（single-turn to multi-turn inclusive language expansion via ChatGPT），该数据集含有56k个多轮对话，其对话主题、词汇和篇章语义更加丰富多样，更加符合在长程多轮对话的应用场景。

#### 偏好数据集

* CValues
  
  * 地址：https://github.com/X-PLUG/CValues
    ![](https://img.shields.io/github/stars/X-PLUG/CValues.svg)
  * 数据集说明：该项目开源了数据规模为145k的价值对齐数据集，该数据集对于每个prompt包括了拒绝&正向建议 (safe and reponsibility) > 拒绝为主(safe) > 风险回复(unsafe)三种类型，可用于增强SFT模型的安全性或用于训练reward模型。

* GPT-4-LLM
  
  * 地址：https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
    ![](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM.svg)
  * 数据集说明：该项目开源了由GPT4生成的多种数据集，包括通过GPT4生成的中英PPO数据，可以用于奖励模型的训练。

* zhihu_rlhf_3k
  
  * 地址：https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k
  * 数据集说明：该项目开源了3k+条基于知乎问答的人类偏好数据集，每个实际的知乎问题下给出了赞同数据较高（chosen）和较低（rejected）的回答，可以用于奖励模型的训练。

* hh_rlhf_cn
  
  * 地址：https://huggingface.co/datasets/dikw/hh_rlhf_cn
  * 数据集说明：基于Anthropic论文Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback 开源的helpful 和harmless数据，使用翻译工具进行了翻译。

* chatbot_arena_conversations
  
  * 地址：https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
  * 数据集说明：该偏好数据集包含20个LLM的输出，其中包括GPT-4和Claude-v1等更强的LLM，它还包含这些最先进模型的许多失败案例。包含来自超过13K个用户的无限制对话。

* UltraFeedback
  
  * 地址：https://github.com/OpenBMB/UltraFeedback
    ![](https://img.shields.io/github/stars/OpenBMB/UltraFeedback.svg)
  * 数据集说明：该数据集是一个大规模、细粒度、多样化的偏好数据集，用于训练强大的奖励模型和批评者模型。该工作从各种资源（包括UltraChat、ShareGPT、Evol-Instruct、TruthfulQA、FalseQA和FLAN，数据集统计数据请参见此处）中收集了约64k条提示。然后使用这些提示来查询多个LLM（模型列表请参见此处），并为每个提示生成4个不同的回复，从而得到总共256k个样本。

### 4. LLM训练微调框架

* DeepSpeed Chat：
  
  * 地址：https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
    ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg)
  * 简介：该项目提供了一键式RLHF训练框架，只需一个脚本即可实现多个训练步骤，包括SFT，奖励模型微调和基于人类反馈的强化学习（RLHF），此外还实现了DeepSpeed HE，统一的高效混合引擎，达到训练和推理引擎之间的过渡是无缝的。

* LLaMA Efficient Tuning：
  
  * 地址：https://github.com/hiyouga/LLaMA-Efficient-Tuning
    ![](https://img.shields.io/github/stars/hiyouga/LLaMA-Efficient-Tuning.svg)
  * 简介：该项目提供了易于使用的基于PEFT的LLaMA微调框架，实现了包括全参数，LoRA，QLoRA等的预训练，指令微调和RLHF，并支持LLaMA, BLOOM, Falcon, Baichuan, InternLM等底座模型。

* ChatGLM Efficient Tuning：
  
  * 地址：https://github.com/hiyouga/ChatGLM-Efficient-Tuning
    ![](https://img.shields.io/github/stars/hiyouga/ChatGLM-Efficient-Tuning.svg)
  * 简介：该项目提供了基于PEFT的高效ChatGLM微调，支持LoRA，P-Tuning V2，全参数微调等模式，并适配了多个微调数据集。

* bert4torch：
  
  * 地址：https://github.com/Tongjilibo/bert4torch
    ![](https://img.shields.io/github/stars/Tongjilibo/bert4torch.svg)
  * 简介：该项目提供了一个大模型的训练和部署框架，包含了目前主要的开源大模型，llama系列，chatglm，bloom系列等等，同时还给出了预训练和微调的示例。

### 5. LLM推理部署框架

* vLLM：
  
  * 地址：https://github.com/vllm-project/vllm
    ![](https://img.shields.io/github/stars/vllm-project/vllm.svg)
  * 简介：适用于大批量Prompt输入，并对推理速度要求高的场景。吞吐量比HuggingFace Transformers高14x-24倍，比HuggingFace Text Generation Inference（TGI）高2.2x-2.5倍，实现了Continuous batching和PagedAttention等技巧。但该框架对适配器（LoRA、QLoRA等）的支持不友好且缺少权重量化。

* DeepSpeed-MII：
  
  * 地址：https://github.com/microsoft/DeepSpeed-MII
    ![](https://img.shields.io/github/stars/microsoft/DeepSpeed-MII.svg)
  * 简介：支持多个机器之间的负载均衡，支持不同的模型库（如Hugging Face、FairSeq等），支持模型量化推理。

* text-generation-inference：
  
  * 地址：https://github.com/huggingface/text-generation-inference
    ![](https://img.shields.io/github/stars/huggingface/text-generation-inference.svg)
  * 简介：用于文本生成推断的Rust、Python和gRPC部署框架，可以监控服务器负载，实现了flash attention和Paged attention，所有的依赖项都安装在Docker中：支持HuggingFace模型；但该框架对适配器（LoRA、QLoRA等）的支持不友好。

* CTranslate2
  
  * 地址：https://github.com/OpenNMT/CTranslate2
    ![](https://img.shields.io/github/stars/OpenNMT/CTranslate2.svg)
  * 简介：基于C++和python的推理框架，支持在CPU和GPU上并行和异步执行，且支持prompt缓存及量化。但缺少对适配器（LoRA、QLoRA等）的支持。

* OpenLLM
  
  * 地址：https://github.com/bentoml/OpenLLM
    ![](https://img.shields.io/github/stars/bentoml/OpenLLM.svg)
  * 简介：支持将要部署的LLM连接多个适配器，可以实现只使用一个底座模型来执行多个特定的任务；支持量化推理和LangChain集成。但对批处理和分布式推理的支持相对不友好。

* MLC LLM
  
  * 地址：https://github.com/mlc-ai/mlc-llm
    ![](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg)
  * 简介：支持不同平台上的不同设备部署推理，包括移动设备（iOS或Android设备等）的高效推理，压缩等。但对大规模批量调用相对不友好。

* LightLLM：
  
  * 地址：https://github.com/ModelTC/lightllm
    ![](https://img.shields.io/github/stars/ModelTC/lightllm.svg)
  * 简介：一个基于 Python 的 LLM（大型语言模型）推理和服务框架，该框架采用轻量级设计、易于扩展和高速性能，LightLLM引入了一种更细粒度的kv cache管理算法 TokenAttention，并设计了一个与TokenAttention高效配合的Efficient Router调度实现。在TokenAttention 和 Efficient Router的相互作用下，LightLLM在大部分场景下都能获得比vLLM 和 Text Generation Inference 得到更高的吞吐，部分场景下可以得到4倍左右的性能提升。

* AirLLM：
  
  * 地址：https://github.com/lyogavin/Anima/tree/main/air_llm
    ![](https://img.shields.io/github/stars/lyogavin/Anima.svg)
  * 简介：该项目开源了一个优化inference内存的推理框架，可实现4GB单卡GPU可以运行70B大语言模型推理。不需要任何损失模型性能的量化和蒸馏，剪枝等模型压缩，该项目采用了分层推理的技术以在较低的内存下实现大模型推理。

* LMDeploy:
  
  * 地址：https://github.com/InternLM/lmdeploy
    ![](https://img.shields.io/github/stars/InternLM/lmdeploy.svg)
  * 简介：该项目支持 LLM（大语言模型）和 VL（视觉语言模型）任务在 NVIDIA 设备上量化、推理和服务。LMDeploy 支持有状态的推理，可以缓存对话，记住历史。它实现了 Persistent Batch(即 Continuous Batch)，Blocked K/V Cache，动态拆分和融合，张量并行，高效的计算 kernel等重要特性。推理性能是 vLLM 的 1.8 倍以上。其 4bit 量化模型推理性能达 FP16 的 2.4 倍以上。

### 6. <a name='LLM评测'></a>LLM评测

* FlagEval （天秤）大模型评测体系及开放平台
  
  * 地址：https://github.com/FlagOpen/FlagEval
    ![](https://img.shields.io/github/stars/FlagOpen/FlagEval.svg)
  * 简介：旨在建立科学、公正、开放的评测基准、方法、工具集，协助研究人员全方位评估基础模型及训练算法的性能，同时探索利用AI方法实现对主观评测的辅助，大幅提升评测的效率和客观性。FlagEval （天秤）创新构建了“能力-任务-指标”三维评测框架，细粒度刻画基础模型的认知能力边界，可视化呈现评测结果。

* C-Eval: 构造中文大模型的知识评估基准：
  
  * 地址：https://github.com/SJTU-LIT/ceval
    ![](https://img.shields.io/github/stars/SJTU-LIT/ceval.svg)
  * 简介：构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。此外还给出了当前主流中文LLM的评测结果。

* OpenCompass:
  
  * 地址：https://github.com/InternLM/opencompass
    ![](https://img.shields.io/github/stars/InternLM/opencompass.svg)
  * 简介：由上海AI实验室发布的面向大模型评测的一站式平台。主要特点包括：开源可复现；全面的能力维度：五大维度设计，提供 50+ 个数据集约 30 万题的的模型评测方案；丰富的模型支持：已支持 20+ HuggingFace 及 API 模型；分布式高效评测：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测；多样化评测范式：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板；灵活化拓展。

* SuperCLUElyb: SuperCLUE琅琊榜
  
  * 地址：https://github.com/CLUEbenchmark/SuperCLUElyb
    ![](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUElyb.svg)
  * 简介：中文通用大模型匿名对战评价基准，这是一个中文通用大模型对战评价基准，它以众包的方式提供匿名、随机的对战。他们发布了初步的结果和基于Elo评级系统的排行榜。

* GAOKAO-Bench:
  
  * 地址：https://github.com/OpenLMLab/GAOKAO-Bench
    ![](https://img.shields.io/github/stars/OpenLMLab/GAOKAO-Bench.svg)
  * 简介：GAOKAO-bench是一个以中国高考题目为数据集，测评大模型语言理解能力、逻辑推理能力的测评框架，收集了2010-2022年全国高考卷的题目，其中包括1781道客观题和1030道主观题，构建起GAOKAO-bench的数据部分。

* AGIEval:
  
  * 地址：https://github.com/ruixiangcui/AGIEval
    ![](https://img.shields.io/github/stars/ruixiangcui/AGIEval.svg)
  * 简介：由微软发布的一项新型基准测试，这项基准选取20种面向普通人类考生的官方、公开、高标准往常和资格考试，包括普通大学入学考试（中国高考和美国 SAT 考试）、法学入学考试、数学竞赛、律师资格考试、国家公务员考试等等。

* Xiezhi:
  
  * 地址：https://github.com/mikegu721/xiezhibenchmark
    ![](https://img.shields.io/github/stars/mikegu721/xiezhibenchmark.svg)
  * 简介：由复旦大学发布的一个综合的、多学科的、能够自动更新的领域知识评估Benchmark，包含了哲学、经济学、法学、教育学、文学、历史学、自然科学、工学、农学、医学、军事学、管理学、艺术学这13个学科门类，24万道学科题目，516个具体学科，249587道题目。

* Open LLM Leaderboard：
  
  * 地址：https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
  * 简介：由HuggingFace组织的一个LLM评测榜单，目前已评估了较多主流的开源LLM模型。评估主要包括AI2 Reasoning Challenge, HellaSwag, MMLU, TruthfulQA四个数据集上的表现，主要以英文为主。

* CMMLU：
  
  * 地址：https://github.com/haonan-li/CMMLU
    ![](https://img.shields.io/github/stars/haonan-li/CMMLU.svg)
  * 简介：CMMLU是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。它包括：需要计算和推理的自然科学，需要知识的人文科学和社会科学,以及需要生活常识的中国驾驶规则等。此外，CMMLU中的许多任务具有中国特定的答案，可能在其他地区或语言中并不普遍适用。因此是一个完全中国化的中文测试基准。

* MMCU：
  
  * 地址：https://github.com/Felixgithub2017/MMCU
    ![](https://img.shields.io/github/stars/Felixgithub2017/MMCU.svg)
  * 简介：该项目提供对中文大模型语义理解能力的测试，评测方式、评测数据集、评测记录都公开，确保可以复现。该项目旨在帮助各位研究者们评测自己的模型性能，并验证训练策略是否有效。

* chinese-llm-benchmark：
  
  * 地址：https://github.com/jeinlee1991/chinese-llm-benchmark
    ![](https://img.shields.io/github/stars/jeinlee1991/chinese-llm-benchmark.svg)
  * 简介：中文大模型能力评测榜单：覆盖百度文心一言、chatgpt、阿里通义千问、讯飞星火、belle / chatglm6b 等开源大模型，多维度能力评测。不仅提供能力评分排行榜，也提供所有模型的原始输出结果！

* Safety-Prompts：
  
  * 地址：https://github.com/thu-coai/Safety-Prompts
    ![](https://img.shields.io/github/stars/thu-coai/Safety-Prompts.svg)
  * 简介：由清华大学提出的一个关于LLM安全评测benchmark，包括安全评测平台等，用于评测和提升大模型的安全性，囊括了多种典型的安全场景和指令攻击的prompt。

* PromptCBLUE: 中文医疗场景的LLM评测基准
  
  * 地址：https://github.com/michael-wzhu/PromptCBLUE
    ![](https://img.shields.io/github/stars/michael-wzhu/PromptCBLUE.svg)
  * 简介：为推动LLM在医疗领域的发展和落地，由华东师范大学联合阿里巴巴天池平台，复旦大学附属华山医院，东北大学，哈尔滨工业大学（深圳），鹏城实验室与同济大学推出PromptCBLUE评测基准, 将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准。

* HalluQA: 中文幻觉评估基准
  
  * 地址：https://github.com/xiami2019/HalluQA
    ![](https://img.shields.io/github/stars/xiami2019/HalluQA.svg)
  * 简介：该项目提出了一个名为HalluQA的基准测试，用于衡量中文大型语言模型中的幻觉现象。HalluQA包含450个精心设计的对抗性问题，涵盖多个领域，并考虑了中国历史文化、风俗和社会现象。在构建HalluQA时，考虑了两种类型的幻觉：模仿性虚假和事实错误，并基于GLM-130B和ChatGPT构建对抗性样本。为了评估，设计了一种使用GPT-4进行自动评估的方法，判断模型输出是否是幻觉。

### 7. <a name='LLM教程'></a>LLM教程

#### LLM基础知识

* HuggingLLM：
  
  * 地址：https://github.com/datawhalechina/hugging-llm
    ![](https://img.shields.io/github/stars/datawhalechina/hugging-llm.svg)
  * 简介：介绍 ChatGPT 原理、使用和应用，降低使用门槛，让更多感兴趣的非NLP或算法专业人士能够无障碍使用LLM创造价值。

* LLMsPracticalGuide：
  
  * 地址：https://github.com/Mooler0410/LLMsPracticalGuide
    ![](https://img.shields.io/github/stars/Mooler0410/LLMsPracticalGuide.svg)
  * 简介：该项目提供了关于LLM的一系列指南与资源精选列表，包括LLM发展历程、原理、示例、论文等。

#### 提示工程教程

* 面向开发者的 LLM 入门课程：
  
  * 地址：https://github.com/datawhalechina/prompt-engineering-for-developers
    ![](https://img.shields.io/github/stars/datawhalechina/prompt-engineering-for-developers.svg)
  * 简介：一个中文版的大模型入门教程，围绕吴恩达老师的大模型系列课程展开，主要包括：吴恩达《ChatGPT Prompt Engineering for Developers》课程中文版，吴恩达《Building Systems with the ChatGPT API》课程中文版，吴恩达《LangChain for LLM Application Development》课程中文版等。

* 提示工程指南:
  
  * 地址：https://www.promptingguide.ai/zh
  * 简介：该项目基于对大语言模型的浓厚兴趣，编写了这份全新的提示工程指南，介绍了大语言模型相关的论文研究、学习指南、模型、讲座、参考资料、大语言模型能力以及与其他与提示工程相关的工具。

* awesome-chatgpt-prompts-zh：
  
  * 地址：https://github.com/PlexPt/awesome-chatgpt-prompts-zh
    ![](https://img.shields.io/github/stars/PlexPt/awesome-chatgpt-prompts-zh.svg)
  * 简介：该项目是ChatGPT中文调教指南。包括各种场景使用指南，让chatgpt知道怎么听你的话，对指令构造可以提供一些参考。

#### LLM应用教程

* LangChain 🦜️🔗 中文网，跟着LangChain一起学LLM/GPT开发：
  
  * 地址：https://www.langchain.asia
  * 简介：Langchain的中文文档，由是两个在LLM创业者维护，希望帮助到从刚进入AI应用开发的朋友们。

* OpenAI Cookbook：
  
  * 地址：https://github.com/openai/openai-cookbook
    ![](https://img.shields.io/github/stars/openai/openai-cookbook.svg)
  * 简介：该项目是OpenAI提供的使用OpenAI API的示例和指导，其中包括如何构建一个问答机器人等教程，能够为从业人员开发类似应用时带来指导。

* 构筑大语言模型应用：应用开发与架构设计：
  
  * 地址：https://github.com/phodal/aigc
    ![](https://img.shields.io/github/stars/phodal/aigc.svg)
  * 简介：该项目开源了一本关于 LLM 在真实世界应用的开源电子书，介绍了大语言模型的基础知识和应用，以及如何构建自己的模型。其中包括Prompt的编写、开发和管理，探索最好的大语言模型能带来什么，以及LLM应用开发的模式和架构设计。

#### LLM实战教程

* LLMs九层妖塔：
  
  * 地址：https://github.com/km1994/LLMsNineStoryDemonTower
    ![](https://img.shields.io/github/stars/km1994/LLMsNineStoryDemonTower.svg)
  * 简介：ChatGLM、Chinese-LLaMA-Alpaca、MiniGPT-4、FastChat、LLaMA、gpt4all等实战与经验。

* llm-action：
  
  * 地址：https://github.com/liguodongiot/llm-action
    ![](https://img.shields.io/github/stars/liguodongiot/llm-action.svg)
  * 简介：该项目提供了一系列LLM实战的教程和代码，包括LLM的训练、推理、微调以及LLM生态相关的一些技术文章等。

* llm大模型训练专栏：
  
  * 地址：https://www.zhihu.com/column/c_1252604770952642560
  * 简介：该项目提供了一系列LLM前言理论和实战实验，包括论文解读与洞察分析。

* 书生·浦语大模型实战营
  
  * 地址：https://github.com/InternLM/tutorial
  * 简介：该课程由上海人工智能实验室重磅推出。课程包括大模型微调、部署与评测全链路，目的是为广大开发者搭建大模型学习和实践开发的平台。
  
  ### 8. <a name='相关仓库'></a>相关仓库

* FindTheChatGPTer：
  
  * 地址：https://github.com/chenking2020/FindTheChatGPTer
    ![](https://img.shields.io/github/stars/chenking2020/FindTheChatGPTer.svg)
  * 简介：ChatGPT爆火，开启了通往AGI的关键一步，本项目旨在汇总那些ChatGPT的开源平替们，包括文本大模型、多模态大模型等，为大家提供一些便利。

* LLM_reviewer：
  
  * 地址：https://github.com/SpartanBin/LLM_reviewer
    ![](https://img.shields.io/github/stars/SpartanBin/LLM_reviewer.svg)
  * 简介：总结归纳近期井喷式发展的大语言模型，以开源、规模较小、可私有化部署、训练成本较低的‘小羊驼类’模型为主。

* Awesome-AITools：
  
  * 地址：https://github.com/ikaijua/Awesome-AITools
    ![](https://img.shields.io/github/stars/ikaijua/Awesome-AITools.svg)
  * 简介：收藏整理了AI相关的实用工具、评测和相关文章。

* open source ChatGPT and beyond：
  
  * 地址：https://github.com/SunLemuria/open_source_chatgpt_list
    ![](https://img.shields.io/github/stars/SunLemuria/open_source_chatgpt_list.svg)
  * 简介：This repo aims at recording open source ChatGPT, and providing an overview of how to get involved, including: base models, technologies, data, domain models, training pipelines, speed up techniques, multi-language, multi-modal, and more to go.

* Awesome Totally Open Chatgpt：
  
  * 地址：https://github.com/nichtdax/awesome-totally-open-chatgpt
    ![](https://img.shields.io/github/stars/nichtdax/awesome-totally-open-chatgpt.svg)
  * 简介：This repo record a list of totally open alternatives to ChatGPT.

* Awesome-LLM：
  
  * 地址：https://github.com/Hannibal046/Awesome-LLM
    ![](https://img.shields.io/github/stars/Hannibal046/Awesome-LLM.svg)
  * 简介：This repo is a curated list of papers about large language models, especially relating to ChatGPT. It also contains frameworks for LLM training, tools to deploy LLM, courses and tutorials about LLM and all publicly available LLM checkpoints and APIs.

* DecryptPrompt：
  
  * 地址：https://github.com/DSXiangLi/DecryptPrompt
    ![](https://img.shields.io/github/stars/DSXiangLi/DecryptPrompt.svg)
  * 简介：总结了Prompt&LLM论文，开源数据&模型，AIGC应用。

* Awesome Pretrained Chinese NLP Models：
  
  * 地址：https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models
    ![](https://img.shields.io/github/stars/lonePatient/awesome-pretrained-chinese-nlp-models.svg)
  * 简介：收集了目前网上公开的一些高质量中文预训练模型。

* ChatPiXiu：
  
  * 地址：https://github.com/catqaq/ChatPiXiu
    ![](https://img.shields.io/github/stars/catqaq/ChatPiXiu.svg)
  * 简介：该项目旨在打造全面且实用的ChatGPT模型库和文档库。当前V1版本梳理了包括：相关资料调研+通用最小实现+领域/任务适配等。

* LLM-Zoo：
  
  * 地址：https://github.com/DAMO-NLP-SG/LLM-Zoo
    ![](https://img.shields.io/github/stars/DAMO-NLP-SG/LLM-Zoo.svg)
  * 简介：该项目收集了包括开源和闭源的LLM模型，具体包括了发布时间，模型大小，支持的语种，领域，训练数据及相应论文/仓库等。

* LLMs-In-China：
  
  * 地址：https://github.com/wgwang/LLMs-In-China
    ![](https://img.shields.io/github/stars/wgwang/LLMs-In-China.svg)
  * 简介：该项目旨在记录中国大模型发展情况，同时持续深度分析开源开放的大模型以及数据集的情况。

* BMList：
  
  * 地址：https://github.com/OpenBMB/BMList
    ![](https://img.shields.io/github/stars/OpenBMB/BMList.svg)
  * 简介：该项目收集了参数量超过10亿的大模型，并梳理了各个大模型的适用模态、发布的机构、适合的语种，参数量和开源地址、API等信息。

* awesome-free-chatgpt：
  
  * 地址：https://github.com/LiLittleCat/awesome-free-chatgpt
    ![](https://img.shields.io/github/stars/LiLittleCat/awesome-free-chatgpt.svg)
  * 简介：该项目收集了免费的 ChatGPT 镜像网站列表，ChatGPT的替代方案，以及构建自己的ChatGPT的教程工具等。

* Awesome-Domain-LLM：
  
  * 地址：https://github.com/luban-agi/Awesome-Domain-LLM
    ![](https://img.shields.io/github/stars/luban-agi/Awesome-Domain-LLM.svg)
  * 简介：该项目收集和梳理垂直领域的开源模型、数据集及评测基准。

## Star History

<a href="https://star-history.com/#HqWu-HITCS/Awesome-Chinese-LLM&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HqWu-HITCS/Awesome-Chinese-LLM&type=Date" />
  </picture>
</a>
</details> 

 <p align="center">
</p>
<details> 
 <summary>ℹ️ <i>Important References </i></summary>


# Important References

- [Guide to Vision-Language Models (VLMs) by Görkem Polat](https://encord.com/blog/vision-language-models-guide/)
- [VLM Primer by Aman Chadha](https://aman.ai/primers/ai/VLM/#google_vignette)
- [Generalized Visual Language Models by Lilian Weng](https://lilianweng.github.io/posts/2022-06-09-vlm/)
- [awesome-vlm-architectures by gokayfem] gokayfem
