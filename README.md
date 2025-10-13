# Data Center Inventory Extractor

🤖 AI-powered tool to extract equipment inventory from data center rack diagrams using state-of-the-art vision-language models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## Features

✨ **Multiple AI Models**
- **Qwen2-VL-7B** - Best for technical diagrams & OCR (Recommended)
- **LLaVA-v1.6-Mistral** - Excellent instruction following
- **PaliGemma** - Lightweight & efficient
- **BLIP-2 Flan-T5-XL** - Good general performance
- **Florence-2** - Fast OCR extraction

🎯 **Key Capabilities**
- Extract equipment lists from PDF diagrams
- OCR text extraction
- Multiple prompt templates
- Model caching for fast reloading
- Real-time VRAM/RAM monitoring
- Batch processing support

## System Requirements

### Minimum Requirements
- **GPU:** NVIDIA RTX 2060 or better (6GB+ VRAM)
- **RAM:** 16GB system RAM
- **Storage:** 50GB free space (for model cache)
- **OS:** Windows 10/11, Linux, macOS

### Recommended Setup
- **GPU:** NVIDIA RTX 3080 (10GB VRAM) or better
- **RAM:** 32GB+ system RAM
- **Storage:** 100GB+ free space
- **CUDA:** 11.8 or newer

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/data-center-inventory-extractor.git
cd data-center-inventory-extractor
