# Data Center Inventory Extractor

🤖 **AI-powered tool to extract equipment inventory from data center rack diagrams using state-of-the-art vision-language models.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-orange.svg)](https://huggingface.co/transformers/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [AI Models](#ai-models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This tool automates the extraction of equipment inventory from data center rack elevation diagrams. Using advanced vision-language models, it can:

- 📊 **Identify equipment** - Recognize servers, switches, PDUs, and other rack-mounted equipment
- 📝 **Extract text** - Read labels, model numbers, and rack unit positions
- 🔗 **Map connections** - Detect cable connections and network topology
- ⚡ **Process at scale** - Handle multi-page PDFs with high-resolution rendering

Perfect for data center audits, inventory management, migration planning, and documentation.

---

## ✨ Features

### 🤖 Multiple AI Models

| Model | VRAM | Best For | Quality | Speed |
|-------|------|----------|---------|-------|
| **Qwen2-VL-7B** ⭐ | 8-9 GB | Technical diagrams, detailed OCR | ⭐⭐⭐⭐⭐ | Medium |
| **LLaVA-v1.6-Mistral** | 8-9 GB | Complex instructions, reasoning | ⭐⭐⭐⭐⭐ | Medium |
| **PaliGemma-3B** | 4-5 GB | Limited VRAM, balanced performance | ⭐⭐⭐⭐ | Fast |
| **BLIP-2 Flan-T5-XL** | 7-8 GB | General descriptions | ⭐⭐⭐ | Fast |
| **Florence-2-Large** | 2-3 GB | Pure OCR, text extraction | ⭐⭐⭐⭐⭐ | Very Fast |

### 🎯 Key Capabilities

- ✅ **PDF Processing** - High-quality rendering at 72-1200 DPI
- ✅ **Dual Resolution System** - Fast display + high-res analysis
- ✅ **Smart Caching** - Models cached locally for instant reloading
- ✅ **Memory Management** - Real-time VRAM/RAM monitoring
- ✅ **Flexible Prompts** - Pre-built templates + custom prompts
- ✅ **OCR Fallback** - Tesseract integration for pure text extraction
- ✅ **Batch Ready** - Process multiple pages efficiently
- ✅ **Export Results** - Download as text files

### 🎨 Advanced Image Processing

- **Multi-DPI Support**: 72, 150, 300, 600, 1200 DPI
- **Color Modes**: RGB or Grayscale
- **Enhancements**: Auto-contrast and sharpening
- **Quality Presets**: Low/Standard/High/Very High/Ultra

---

## 🎬 Demo

### Input: Data Center Rack Diagram
![Sample Input](docs/sample_input.png)

### Output: Structured Equipment List
RACK LAYOUT (Top to Bottom):

Position U0-U2:

Equipment: DIAL ALARM01 (Audio Alarm Panel)
Type: AUD ALM PNL
Position D0:

Equipment: DC-AC INVERTER
Connections: TO CSPCO, TO DF
Position D1:

Equipment: MISC CAB
Contains: (2) 16A RA, (1) TBCU
Position D2:

Equipment: MISC CAB
Contains: (3) 3A SMSI TRANS
...


---

## 💻 System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 2060 (6GB VRAM) or better |
| **CPU** | Intel i5 / AMD Ryzen 5 (CPU-only mode) |
| **RAM** | 16GB system RAM |
| **Storage** | 50GB free space (for model cache) |
| **OS** | Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+ |
| **CUDA** | 11.8 or newer (for GPU acceleration) |

### Recommended Setup

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 3080 (10GB VRAM) or RTX 4070 |
| **CPU** | Intel i7 / AMD Ryzen 7 |
| **RAM** | 32GB+ system RAM |
| **Storage** | 100GB+ NVMe SSD |
| **CUDA** | 12.1+ |

### CPU-Only Mode

Works without GPU but 5-10x slower:
- **RAM**: 16GB minimum, 32GB recommended
- **Swap**: 16GB virtual memory recommended
- **Models**: Florence-2 or PaliGemma recommended

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/youcangetjules/Lumen.git

cd data-center-inventory-extractor
```

### 2. Create Virtual Environment
#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch
   
#### For GPU (CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For GPU (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU-only:

```bash
pip install torch torchvision
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Install Tesseract OCR

#### Windows:

Download from: 
```bash
https://github.com/UB-Mannheim/tesseract/wiki
```

Run installer (use default path: C:\Program Files\Tesseract-OCR)
App will auto-detect on startup

#### Linux:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS:

```bash
brew install tesseract
```

### 6. Verify Installation

#### Check CUDA availability

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Check GPU info (if available)
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('CPU mode')"
```

# ⚡ Quick Start

Basic Usage

### Run the application


```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

#### First-Time Setup

Upload a PDF - Click "Browse files" and select your rack diagram

Choose Quality - Select DPI (300 DPI recommended) << Go higher if needs be, but with an RTX 3080 - this becomes SLOW.

Select Model - Qwen2-VL-7B recommended for best quality. I am busy working on how to split up text and graphics (topology) into separate chucks - future feature

Pick Template - Use "Detailed Inventory" for comprehensive extraction. It accepts free text.

Analyze - Click "Analyze with AI" (first run downloads model ~15GB) these models are cached on your local machine - all about 10+ GB so if you're limited on space you can delete the chache (and start again...)

# 📖 Usage Guide

## 1. Upload & Configure

### Supported Formats

- PDF (single or multi-page)
- **Recommended:** Vector PDFs or high-quality scans

### Quality Settings

| DPI | Description | Recommendation |
|-----|-------------|----------------|
| **72** | Fast preview, low quality | Quick checks only |
| **150** | Standard documents | General documents |
| **300** ⭐ | Best balance | **Recommended starting point** |
| **600** | Tiny text, detailed diagrams | **Best for technical diagrams with GPU** |
| **1200** | Ultra high-res | Archival quality (very slow) |

> **Note:** 600 DPI is probably the best starting point for technical diagrams, but make sure you have a powerful GPU!

---

## 2. Choose Display Mode

### Fit to Width
- Scales image to browser width
- Fast, convenient
- May appear blurry at high DPI

### Actual Size
- Pixel-perfect rendering
- Requires scrolling for large images
- Sharp and clear

---

## 3. Select AI Model

### For Technical Diagrams
- ✅ **Qwen2-VL-7B** (best OCR accuracy)
- ✅ **Florence-2** (fastest pure OCR)

### For Complex Analysis
- ✅ **LLaVA-v1.6-Mistral** (best reasoning)
- ✅ **Qwen2-VL-7B** (detailed descriptions)

### For Limited VRAM (<8GB)
- ✅ **PaliGemma-3B** (4-5GB VRAM)
- ✅ **Florence-2** (2-3GB VRAM)

> **💡 Tip:** If you're running into VRAM issues, ask your boss for a better computer! 🚀

---

## 4. Craft Your Prompt

### Built-in Templates

| Template | Best For | Output Format |
|----------|----------|---------------|
| **Detailed Inventory** | Complete equipment list | Structured list with specs |
| **Structured Table** | Database import | Markdown table |
| **Technical Diagram** | System documentation | Categorized sections |
| **Quick List** | Fast overview | Simple enumeration |
| **OCR Extract** | Raw text extraction | Unstructured text |

### Custom Prompts Examples

```bash
List all network equipment with IP addresses and connection details
```

```bash
Create a CSV-compatible inventory with columns: Position, Type, Model, Serial
```

```bash
Identify all power distribution units and their amp ratings
```

---

## 5. Analyze & Export

- **Processing time:** ~5 minutes on average (GPU at 600 DPI)
- Results displayed in markdown format
- Download as `.txt` file
- Copy/paste to documentation

---

## 🤖 AI Models

> **📌 READ THIS SECTION:** It will save you time depending on the task!

### Qwen2-VL-7B-Instruct ⭐ Recommended

**Best for:** Technical diagrams, equipment lists, detailed OCR

**Strengths:**
- Excellent text recognition (labels, model numbers, part codes)
- Understands technical terminology
- Accurate position/rack unit detection
- Good connection mapping

**VRAM:** 8-9 GB | **Speed:** Medium

**Example Output:**

RACK LAYOUT (U0 to U42):

U0-U2: DIALAMM01

Type: Audio Alarm Panel (AUD ALM PNL)
Purpose: Alarm monitoring and notification
U3-U5: MISC CAB

Equipment 1: Reserve Panel (RES PNL)
Equipment 2: MDF 1ST TRK
Equipment 3: TBCU (2 units)

### LLaVA-v1.6-Mistral-7B

**Best for:** Complex instructions, detailed reasoning

**Strengths:**
- Excellent instruction following
- Natural language understanding
- Detailed explanations
- Good spatial reasoning

**VRAM:** 8-9 GB | **Speed:** Medium

### PaliGemma-3B-Mix-448

**Best for:** Limited VRAM, balanced performance

**Strengths:**
- Memory efficient (4-5GB VRAM)
- Fast inference
- Good general performance
- Lightweight deployment

**VRAM:** 4-5 GB | **Speed:** Fast

### Florence-2-Large

**Best for:** Pure OCR, text extraction, object detection

**Strengths:**
- Extremely fast
- Low memory footprint
- Excellent text recognition
- Built-in OCR tasks

**VRAM:** 2-3 GB | **Speed:** Very Fast

### BLIP-2 Flan-T5-XL

**Best for:** General descriptions, question answering

**Strengths:**
- Good general understanding
- Natural language generation
- Stable performance

**VRAM:** 7-8 GB | **Speed:** Fast

---

## ⚙️ Configuration

### DPI Selection Guide

| DPI | Use Case | File Size | Processing Time |
|-----|----------|-----------|-----------------|
| **72** | Quick preview | Small | Very Fast |
| **150** | Standard documents | Medium | Fast |
| **300** ⭐ | Technical diagrams | Large | Medium |
| **600** | Tiny text (<8pt font) | Very Large | Slow |
| **1200** | Archival quality | Huge | Very Slow |

### Color Mode

**RGB (Color):**
- Full color preservation
- Better for color-coded diagrams
- Larger file size

**Grayscale:**
- Can improve OCR accuracy
- Smaller file size
- Faster processing

### Enhancement Options

**Enhance Contrast:**
- ✅ Recommended for scanned documents
- Improves text visibility
- May over-enhance clean diagrams

**Sharpen Image:**
- ✅ Recommended for soft/blurry images
- Makes text crisper
- May amplify noise

---

# 🔧 Troubleshooting

## Common Issues

### 1. CUDA Not Available

**Problem:** App shows "CUDA Not Available"

**Solution:**

## Check if NVIDIA GPU is detected
```bash
nvidia-smi
```

## Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

##2. Out of Memory (VRAM)
Problem: Error during model loading or inference

#### Solutions:

Switch to CPU mode (slower but works)
Use a smaller model (Florence-2 or PaliGemma)
Close other GPU applications
Clear GPU cache (button in sidebar)
Reduce DPI setting

## 3. Out of Memory (RAM)
Problem: System freezes or crashes

#### Solutions:

Close other applications
Use Florence-2 (smallest model)
Increase virtual memory/swap space
Upgrade system RAM

## 4. Model Download Fails
Problem: Download interrupted or corrupted

#### Solution:

# Clear cache and retry

```bash
rm -rf ~/.cache/huggingface/hub/*
```

Or use the cache manager button in the app sidebar.

## 5. Tesseract Not Found
Problem: OCR mode fails

#### Windows Solution:

Download installer from: 

```bash
https://github.com/UB-Mannheim/tesseract/wiki
```

Run the installer (use default path: C:\Program Files\Tesseract-OCR)
The app will auto-detect on startup

#### Latest Windows Installer:

tesseract-ocr-w64-setup-5.5.0.20241111.exe (64-bit)

#### Linux Solution:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS Solution:

```bash
brew install tesseract
```

## 6. Poor OCR Quality

### Solutions:

Increase DPI to 600 or 1200
Enable "Enhance Contrast" and "Sharpen"
Use Grayscale mode
Try Qwen2-VL instead (better than Tesseract)

# 🚀 Performance Tips

## For Best Quality
Use 300-600 DPI for analysis
Enable enhancements (contrast + sharpen)
Choose Qwen2-VL or LLaVA for complex diagrams
Use structured prompts (table format works best)

##For Speed
Use 150 DPI for quick processing
Choose Florence-2 (fastest model)
Disable enhancements (faster rendering)
Use GPU mode (10x faster than CPU)

## For Limited VRAM
Use PaliGemma-3B (4-5GB) or Florence-2 (2-3GB)
Switch to CPU mode as fallback
Close other GPU applications
Process one page at a time

## Batch Processing
For multiple PDFs:
#### Process all PDFs in a folder (future feature)
#### Currently: Upload one at a time

## 📊 Model Comparison

### Performance Matrix

| Metric | Qwen2-VL | LLaVA-v1.6 | PaliGemma | Florence-2 | BLIP-2 |
|--------|----------|------------|-----------|------------|--------|
| **Text Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Reasoning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memory Efficient** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Detail Level** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Technical Specifications

| Model | VRAM Usage | Model Size | Inference Speed | Best Use Case |
|-------|-----------|------------|-----------------|---------------|
| **Qwen2-VL-7B** | 8-9 GB | ~15 GB | Medium | Technical diagrams with OCR |
| **LLaVA-v1.6** | 8-9 GB | ~15 GB | Medium | Complex reasoning tasks |
| **PaliGemma-3B** | 4-5 GB | ~6 GB | Fast | Limited VRAM systems |
| **Florence-2** | 2-3 GB | ~1.5 GB | Very Fast | Pure OCR extraction |
| **BLIP-2** | 7-8 GB | ~11 GB | Fast | General descriptions |

### Detailed Comparison

#### Text Recognition Accuracy
- **Best:** Qwen2-VL, Florence-2 (⭐⭐⭐⭐⭐)
- **Good:** LLaVA-v1.6, PaliGemma (⭐⭐⭐⭐)
- **Adequate:** BLIP-2 (⭐⭐⭐)

#### Reasoning & Understanding
- **Best:** Qwen2-VL, LLaVA-v1.6 (⭐⭐⭐⭐⭐)
- **Good:** PaliGemma, BLIP-2 (⭐⭐⭐)
- **Basic:** Florence-2 (⭐⭐)

#### Processing Speed
- **Fastest:** Florence-2 (⭐⭐⭐⭐⭐)
- **Fast:** PaliGemma, BLIP-2 (⭐⭐⭐⭐)
- **Medium:** Qwen2-VL, LLaVA-v1.6 (⭐⭐⭐)

#### Memory Efficiency
- **Most Efficient:** Florence-2 (⭐⭐⭐⭐⭐)
- **Efficient:** PaliGemma (⭐⭐⭐⭐)
- **Standard:** BLIP-2 (⭐⭐⭐)
- **High Usage:** Qwen2-VL, LLaVA-v1.6 (⭐⭐)

### Recommendation Guide

**Choose Qwen2-VL if:**
- You need the best OCR accuracy for technical diagrams
- You have 8GB+ VRAM available
- Processing time is not critical
- You need detailed equipment specifications

**Choose LLaVA-v1.6 if:**
- You need complex reasoning and understanding
- You want detailed natural language descriptions
- You have 8GB+ VRAM available
- You need to follow complex instructions

**Choose PaliGemma if:**
- You have limited VRAM (4-6GB)
- You need balanced performance
- Speed is important
- You want good general-purpose results

**Choose Florence-2 if:**
- You need pure OCR/text extraction
- You have very limited VRAM (2-3GB)
- Speed is the top priority
- You don't need complex reasoning

**Choose BLIP-2 if:**
- You need general image descriptions
- You have moderate VRAM (7-8GB)
- You want stable, predictable results
- You need question-answering capabilities

### Download Times (100 Mbps Connection)

| Model | Size | Download Time | Disk Space Required |
|-------|------|---------------|---------------------|
| Qwen2-VL-7B | ~15 GB | 20-25 minutes | 15 GB |
| LLaVA-v1.6 | ~15 GB | 20-25 minutes | 15 GB |
| BLIP-2 Flan-T5-XL | ~11 GB | 15-20 minutes | 11 GB |
| PaliGemma-3B | ~6 GB | 8-10 minutes | 6 GB |
| Florence-2-Large | ~1.5 GB | 2-3 minutes | 2 GB |

> **💡 Note:** Models are cached locally. First download takes time, but subsequent loads are instant!

### GPU Requirements

| Model | Minimum GPU | Recommended GPU | CPU Fallback |
|-------|-------------|-----------------|--------------|
| **Qwen2-VL** | RTX 2060 (6GB) | RTX 3080 (10GB) | ⚠️ Very Slow |
| **LLaVA-v1.6** | RTX 2060 (6GB) | RTX 3080 (10GB) | ⚠️ Very Slow |
| **PaliGemma** | GTX 1660 (6GB) | RTX 3060 (12GB) | ✅ Workable |
| **Florence-2** | GTX 1650 (4GB) | Any GPU | ✅ Good |
| **BLIP-2** | RTX 2060 (6GB) | RTX 3070 (8GB) | ⚠️ Slow |

---
# 🤝 Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

# Development Setup

# Clone repo

```bash
git clone https://github.com/yourusername/data-center-inventory-extractor.git
cd data-center-inventory-extractor
```

# Install dev dependencies
```bash
pip install -r requirements-dev.txt
```

# Run tests
```bash
pytest tests/
```
# Format code
```bash
black app.py
```

# Lint
```bash
flake8 app.py
```

# Feature Requests
Open an issue with the label enhancement

# Bug Reports
Open an issue with:

System info (OS, GPU, RAM)
Steps to reproduce
Error messages
Screenshots

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 🙏 Acknowledgments

# 📞 Support
Email: julian.garrett@aliniant.com

Under MIT License @ Aliniant Labs 2025

# 🗺️ Roadmap
 Batch PDF processing
 CSV/Excel export
 Database integration
 API endpoint
 Docker deployment
 Web-based annotation tool
 Multi-language support
 Custom model fine-tuning
 
# Made with ❤️ for anyone who find this useful. There's a fair way to go yet!!!
