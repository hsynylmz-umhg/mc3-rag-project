# MC3: Hardware-Optimized RAG Pipeline for AMD GPU

This repository contains the source code and configuration for Mini Challenge 3 of the LabLab x AMD AI Academy Challenge. 

## 📦 Docker Hub Repository
The fully built, hardware-optimized RAG image (approx. 40GB) is compiled and pushed to Docker Hub. You can access it here:
**[hsynylmz/mc3-rag on Docker Hub](https://hub.docker.com/r/hsynylmz/mc3-rag)**

## 🚀 Evaluation & Run Instructions
To evaluate the submission with native AMD GPU acceleration (zero-hallucination, high-throughput), please pull and run the unified image using the following command. This ensures the container has the required `/dev/kfd` and `/dev/dri` hardware access flags:

```bash
docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add=video hsynylmz/mc3-rag:latest
```
## 🛠️ Architecture & Tech Stack

* **Mandated Base Image:** `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
* **Frameworks:** Python, PyTorch, RAG Pipeline
* **Hardware Acceleration:** AMD ROCm (Instinct / Radeon support)
* **Key Features:**
  * Strict adherence to zero-hallucination policy
  * Persistent long-running process architecture
  * Resilient handling of unknown/encrypted corpus files
