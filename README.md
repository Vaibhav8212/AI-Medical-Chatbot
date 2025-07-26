# 🩺 AI Doctor - Intelligent Symptom Checker (Fine-tuned DeepSeek-R1)

This project fine-tunes the **DeepSeek-R1 LLM** on a **medical chain-of-thought (CoT)** reasoning dataset using [Unsloth](https://github.com/unslothai/unsloth). It aims to build an intelligent AI-powered medical assistant that can provide step-by-step reasoning and diagnosis based on complex patient symptoms.

---

## 🚀 Overview

- 🧠 Fine-tunes **DeepSeek-R1** using **LoRA** and **Chain-of-Thought medical dataset**
- 💡 Uses the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset
- ⚡ Based on [Unsloth](https://github.com/unslothai/unsloth) for efficient low-RAM finetuning
- 📈 Monitors training with **Weights & Biases (WandB)** integration
- 🧪 Evaluates the model with inference before and after fine-tuning

---

## 🧰 Tech Stack

- 🤖 **Model**: `DeepSeek-R1-Distill-Llama-8B`
- 🔧 **Frameworks**: `Unsloth`, `Transformers`, `SFTTrainer`, `LoRA`
- 📊 **Monitoring**: `Weights & Biases`
- 📚 **Dataset**: `medical-o1-reasoning-SFT` (from FreedomIntelligence)
- ⚡ **Accelerators**: CUDA / GPU with 4-bit quantization support

---

##   Setup Instructions

### 1.  Hugging Face API Token

1. Store your Hugging Face token securely in Google Colab:

```python
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
```

2. Install Unsloth and Dependencies
   
   ```python
   !pip install unsloth
   !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
   ```

---
## Dataset

Name: FreedomIntelligence/medical-o1-reasoning-SFT

Type: Chain-of-thought CoT formatted medical QA

Split Used: First 500 samples from the train set

---

  ## Finetuning Workflow
   Load and quantize the model using Unsloth

   Apply a structured medical prompt for system reasoning

   Preprocess CoT reasoning samples for fine-tuning

   Fine-tune using SFTTrainer and LoRA

   Log training with Weights & Biases

   Future Scope
   Expand to multiple languages for global healthcare

   Add clinical guideline documents for grounding

   Convert into a mobile or web-based AI doctor chatbot

   Add HIPAA-style anonymization for sensitive queries

   Integrate with EHR systems or diagnostic decision support

   Use Reinforcement Learning from Human Feedback (RLHF) to improve answer quality



