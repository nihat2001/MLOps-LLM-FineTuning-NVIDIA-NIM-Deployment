# 🚀 MLOps-LLM-FineTuning-NVIDIA-NIM-Deployment

| Metric / Status | Details |
| :--- | :--- |
| **Base Model** | Meta Llama-3.1-8B-Instruct |
| **Fine-Tuning Method** | QLoRA (4-bit Quantization, Unsloth, PEFT) |
| **Inference Engine** | NVIDIA NIM Container (`nvcr.io/nim/meta/llama-3.1-8b-instruct`) + vLLM |
| **Cloud Infrastructure** | RunPod (NVIDIA L40S 48GB VRAM) & Google Colab (T4 GPU) |
| **Serving Standard** | OpenAI-Compatible REST API (`/v1/chat/completions`) |

---

## 🏗️ End-to-End Pipeline Overview

| Stage | Component | Technical Action & Responsibilities |
| :--- | :--- | :--- |
| **1. Data & Training** | Google Colab (T4) | • Loaded 4-bit base model via Unsloth<br>• Applied LoRA (`r=16`, `alpha=32`) to projection layers<br>• Applied `train_on_responses_only` loss masking<br>• Formatted dataset with Llama 3.1 Chat Template |
| **2. Artifact Storage** | Hugging Face Hub | • Exported lightweight PEFT LoRA adapter (~160 MB)<br>• Updated `base_model_name_or_path` for NIM compatibility<br>• Stored adapter in public/private repository |
| **3. Infrastructure** | RunPod (L40S GPU) | • Provisioned 48GB VRAM node<br>• Authenticated via NVIDIA NGC Registry (`ngc-nvcr`)<br>• Configured cache path, LoraWatcher, and rank limits |
| **4. Production Serving** | NVIDIA NIM + vLLM | • Initialized base model in frozen state<br>• Dynamically injected LoRA via `/v1/load_lora_adapter`<br>• Opened OpenAI-compliant API endpoint for inference |

---

## 🛠️ Technology Stack

| Domain | Tools & Technologies |
| :--- | :--- |
| **Fine-Tuning Frameworks** | `Unsloth`, `TRL` (SFTTrainer), `PEFT`, `Transformers`, `PyTorch` |
| **Serving & Orchestration** | `NVIDIA NIM`, `vLLM Engine`, `LoraWatcher`, `Docker Container` |
| **Cloud & Registries** | `RunPod Cloud`, `NVIDIA NGC Catalog`, `Hugging Face Hub` |
| **Protocols & Interfaces** | `OpenAI API Protocol`, `REST API`, `cURL`, `Bash / Python` |

---

## ⚡ Technical Highlights

| Feature | Description | MLOps Value |
| :--- | :--- | :--- |
| **4-Bit QLoRA** | Quantized base weights with trainable LoRA adapter matrices. | Reduces VRAM consumption by ~75% while keeping core model performance. |
| **Response Loss Masking** | System & User tokens masked with `label = -100`. | Forces gradient updates strictly on Assistant responses, avoiding redundant learning. |
| **Dynamic Adapter Injection** | NIM `/v1/load_lora_adapter` endpoint integration. | Enables zero-downtime hot-swapping of fine-tuned adapters on a running server. |
| **Enterprise Standardization** | OpenAI API standard wrapper over vLLM. | Allows instant integration with existing LLM clients and orchestrators (e.g., LangChain). |

---

## 🚀 Execution & Command Reference

| Step | Objective | Command / Action |
| :--- | :--- | :--- |
| **1** | **Health Check** | `curl -s $NIM/v1/models \| python3 -m json.tool` |
| **2** | **Load LoRA Adapter** | `curl -X POST $NIM/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "novashop", "lora_path": "USERNAME/novashop-support-lora"}'` |
| **3** | **Confirm Loading** | `curl -s $NIM/v1/models \| python3 -c "import sys,json;print([m['id'] for m in json.load(sys.stdin)['data']])"` |
| **4** | **Query Fine-Tuned Model**| `curl $NIM/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "novashop", "messages": [{"role": "user", "content": "Qaytarma müddəti neçə gündür?"}]}'` |

---

## 🔒 Environment Variables Reference

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `NGC_API_KEY` | `<YOUR_NGC_KEY>` | Authenticates container image & weight pulls from NVIDIA Catalog |
| `NIM_CACHE_PATH` | `/opt/nim/.cache` | Persistent storage path for downloaded model artifacts |
| `NIM_PEFT_SOURCE` | `/opt/nim/.cache/loras` | Local cache directory for dynamic LoRA adapters |
| `NIM_PEFT_REFRESH_INTERVAL` | `30` | LoraWatcher polling interval in seconds |
| `NIM_MAX_LORA_RANK` | `32` | Maximum allowable LoRA rank accepted by the NIM server |
