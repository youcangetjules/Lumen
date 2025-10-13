import streamlit as st
import fitz
from PIL import Image, ImageFilter, ImageEnhance
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
import gc
from pathlib import Path
import shutil
from datetime import datetime
import psutil
import os
import platform

# Optional: Import for advanced progress tracking
from transformers import StoppingCriteria, StoppingCriteriaList

class ProgressStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria with progress callback"""
    def __init__(self, progress_bar, status_text, max_tokens=512):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_tokens = max_tokens
        self.current_tokens = 0
        
    def __call__(self, input_ids, scores, **kwargs):
        self.current_tokens = input_ids.shape[-1]
        progress = min(0.4 + (self.current_tokens / self.max_tokens) * 0.5, 0.95)
        self.progress_bar.progress(progress)
        self.status_text.text(f"🤖 Generating... ({self.current_tokens} tokens)")
        return False

st.set_page_config(page_title="Data Center Inventory Extractor", layout="wide")

st.title("Data Center Inventory Extractor")

APP_VERSION = "8.0"
COMPATIBLE_MODELS = {
    "Salesforce/blip2-opt-2.7b": "3.0",
    "Salesforce/blip2-flan-t5-xl": "3.0",
    "microsoft/Florence-2-large": "3.0",
    "Qwen/Qwen2-VL-7B-Instruct": "3.0",
    "llava-hf/llava-v1.6-mistral-7b-hf": "3.0",
    "google/paligemma-3b-mix-448": "3.0"
}

def create_progress_callback(progress_bar, status_text):
    """Create a callback for generation progress"""
    step = [0]
    
    def callback(input_ids, scores, **kwargs):
        step[0] += 1
        if step[0] % 5 == 0:  # Update every 5 tokens
            progress = min(step[0] / 100, 0.99)  # Cap at 99%
            progress_bar.progress(progress)
            status_text.text(f"Generating... ({step[0]} tokens)")
        return False
    
    return callback

# ============= CUDA DETECTION =============
def check_cuda_setup():
    """Comprehensive CUDA detection"""
    cuda_info = {
        'pytorch_cuda': torch.cuda.is_available(),
        'cuda_version': None,
        'cudnn_version': None,
        'gpu_name': None,
        'gpu_count': 0,
        'compute_capability': None,
        'driver_version': None,
    }
    
    if torch.cuda.is_available():
        cuda_info['cuda_version'] = torch.version.cuda
        cuda_info['cudnn_version'] = torch.backends.cudnn.version()
        cuda_info['gpu_count'] = torch.cuda.device_count()
        cuda_info['gpu_name'] = torch.cuda.get_device_name(0)
        
        # Get compute capability
        capability = torch.cuda.get_device_capability(0)
        cuda_info['compute_capability'] = f"{capability[0]}.{capability[1]}"
        
        # Try to get driver version
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                cuda_info['driver_version'] = result.stdout.strip()
        except:
            pass
    
    return cuda_info

cuda_info = check_cuda_setup()

# Auto-detect Tesseract location for Windows
if platform.system() == 'Windows':
    try:
        import pytesseract
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    except ImportError:
        pass

def get_ram_info():
    """Get system RAM and virtual memory usage"""
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        'ram_total_gb': ram.total / 1024**3,
        'ram_used_gb': ram.used / 1024**3,
        'ram_available_gb': ram.available / 1024**3,
        'ram_percent': ram.percent,
        'swap_total_gb': swap.total / 1024**3,
        'swap_used_gb': swap.used / 1024**3,
        'swap_percent': swap.percent
    }

def get_vram_info():
    """Get GPU VRAM usage"""
    if not torch.cuda.is_available():
        return None
    
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
    vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
    vram_free = vram_total - vram_reserved
    
    return {
        'vram_total_gb': vram_total,
        'vram_allocated_gb': vram_allocated,
        'vram_reserved_gb': vram_reserved,
        'vram_free_gb': vram_free,
        'vram_percent': (vram_reserved / vram_total) * 100
    }

# ============= ENHANCED SYSTEM INFO SIDEBAR =============
st.sidebar.markdown("### 💻 System Info")

# CUDA Status with detailed info
if cuda_info['pytorch_cuda']:
    st.sidebar.success(f"✅ **CUDA Available**")
    st.sidebar.write(f"**GPU:** {cuda_info['gpu_name']}")
    
    with st.sidebar.expander("🔧 CUDA Details", expanded=False):
        st.write(f"**CUDA Version:** {cuda_info['cuda_version']}")
        st.write(f"**cuDNN Version:** {cuda_info['cudnn_version']}")
        st.write(f"**Compute Capability:** {cuda_info['compute_capability']}")
        if cuda_info['driver_version']:
            st.write(f"**Driver Version:** {cuda_info['driver_version']}")
        st.write(f"**GPU Count:** {cuda_info['gpu_count']}")
        
        # Show VRAM
        vram = get_vram_info()
        if vram:
            st.write(f"**VRAM:** {vram['vram_free_gb']:.1f}GB / {vram['vram_total_gb']:.1f}GB free")
            st.progress(vram['vram_percent'] / 100)
else:
    st.sidebar.error("❌ **CUDA Not Available**")
    st.sidebar.caption("Running in CPU-only mode")
    
    with st.sidebar.expander("💡 Enable CUDA", expanded=False):
        st.markdown("""
        **To enable GPU acceleration:**
        
        1. **Check GPU:** Ensure you have an NVIDIA GPU
        2. **Install CUDA Toolkit:**
           - Download from: https://developer.nvidia.com/cuda-downloads
        3. **Install PyTorch with CUDA:**
        ```bash
        pip uninstall torch torchvision
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ```
        4. **Restart app**
        
        **Check NVIDIA driver:**
        ```bash
        nvidia-smi
        ```
        """)

# RAM Info
ram_info = get_ram_info()
st.sidebar.write(f"**RAM:** {ram_info['ram_available_gb']:.1f}GB / {ram_info['ram_total_gb']:.1f}GB free")
st.sidebar.write(f"**RAM Usage:** {ram_info['ram_percent']:.1f}%")
st.sidebar.progress(ram_info['ram_percent'] / 100)

if ram_info['swap_total_gb'] > 0:
    st.sidebar.write(f"**Virtual RAM:** {ram_info['swap_used_gb']:.1f}GB / {ram_info['swap_total_gb']:.1f}GB used")

st.sidebar.markdown("---")

# ============= COMPUTE DEVICE SELECTOR =============
st.sidebar.markdown("### ⚙️ Compute Settings")

# Device selection with dynamic options
if cuda_info['pytorch_cuda']:
    device_options = ["GPU (CUDA)", "CPU Only"]
    default_device = 0  # GPU by default
    help_text = f"Using {cuda_info['gpu_name']} - ~10x faster than CPU"
else:
    device_options = ["CPU Only"]
    default_device = 0
    help_text = "⚠️ GPU not available - models will run slower on CPU"

selected_device = st.sidebar.radio(
    "Compute Device:",
    device_options,
    index=default_device,
    help=help_text
)

# Set device based on selection
use_gpu = selected_device == "GPU (CUDA)" and cuda_info['pytorch_cuda']

if use_gpu:
    st.sidebar.success("🚀 **Using GPU acceleration**")
    device = "cuda"
    vram = get_vram_info()
    if vram:
        st.sidebar.caption(f"VRAM: {vram['vram_free_gb']:.1f}GB free")
else:
    st.sidebar.info("🐌 **Using CPU** (slower)")
    device = "cpu"
    if not cuda_info['pytorch_cuda']:
        st.sidebar.caption("GPU not available on this system")

# Performance warning for CPU mode
if device == "cpu":
    st.sidebar.warning("⚠️ CPU mode is 5-10x slower than GPU")
    st.sidebar.caption("First inference may take 2-5 minutes")

st.sidebar.markdown("---")

# ============= GPU CACHE MANAGEMENT =============
if torch.cuda.is_available():
    st.sidebar.markdown("### 🧹 GPU Cache")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            torch.cuda.empty_cache()
            gc.collect()
            st.rerun()
    
    with col2:
        if st.button("Refresh Stats", use_container_width=True):
            st.rerun()

st.sidebar.markdown("---")

# ============= MODEL CACHE MANAGER =============
def get_cache_dir():
    """Get HuggingFace cache directory"""
    return Path.home() / ".cache" / "huggingface" / "hub"

def get_cached_models():
    """Get list of cached models with compatibility info"""
    cache_dir = get_cache_dir()
    models = []
    
    if not cache_dir.exists():
        return models
    
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("models--"):
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1024**3
            
            compatible = model_name in COMPATIBLE_MODELS
            
            models.append({
                'name': model_name,
                'path': model_dir,
                'size_gb': size,
                'compatible': compatible
            })
    
    return sorted(models, key=lambda x: x['size_gb'], reverse=True)

st.sidebar.markdown("### 💾 Model Cache Manager")
st.sidebar.caption(f"App Version: {APP_VERSION}")

cached_models = get_cached_models()

if cached_models:
    total_size = sum(m['size_gb'] for m in cached_models)
    st.sidebar.write(f"**Total Cache:** {total_size:.1f} GB")
    st.sidebar.write(f"**Models Cached:** {len(cached_models)}")
    
    with st.sidebar.expander("📦 Cached Models", expanded=False):
        for model in cached_models:
            status = "✅" if model['compatible'] else "⚠️"
            st.write(f"{status} **{model['name']}**")
            st.caption(f"Size: {model['size_gb']:.2f} GB")
            
            if not model['compatible']:
                st.caption("⚠️ Incompatible with current app version")
            
            if st.button(f"🗑️ Delete", key=f"del_{model['name']}"):
                try:
                    shutil.rmtree(model['path'])
                    st.success(f"Deleted {model['name']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            st.markdown("---")
    
    if st.sidebar.button("🗑️ Clear All Cache"):
        try:
            cache_dir = get_cache_dir()
            shutil.rmtree(cache_dir)
            st.sidebar.success("Cache cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
else:
    st.sidebar.info("No models cached yet")

st.sidebar.markdown("---")

# ============= MODEL SIZE ESTIMATES =============
def get_model_size(model_name):
    """Estimate model download size"""
    sizes = {
        "Salesforce/blip2-opt-2.7b": "5.4 GB",
        "Salesforce/blip2-flan-t5-xl": "11 GB",
        "microsoft/Florence-2-large": "1.5 GB",
        "Qwen/Qwen2-VL-7B-Instruct": "15 GB",
        "llava-hf/llava-v1.6-mistral-7b-hf": "15 GB",
        "google/paligemma-3b-mix-448": "6 GB"
    }
    return sizes.get(model_name, "Unknown")

# ============= ADVANCED PDF PROCESSING =============
def pdf_to_images(pdf_file, dpi=300, color_mode='RGB', enhance_quality=True, sharpen=True):
    """Convert PDF pages to high-quality images with advanced options
    
    Args:
        pdf_file: Uploaded PDF file
        dpi: Dots per inch (72-1200)
             72  = Low quality (fast)
             150 = Standard quality
             300 = High quality (recommended) ⭐
             600 = Very high quality
             1200 = Ultra high quality (slow, very large)
        color_mode: 'RGB' or 'L' (grayscale)
        enhance_quality: Apply contrast enhancement
        sharpen: Apply sharpening filter for crisp text
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    display_images = []  # For showing in UI at original size
    
    # Calculate zoom factor based on DPI
    # PyMuPDF default is 72 DPI, so zoom = target_dpi / 72
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Original display matrix (for UI preview)
    display_mat = fitz.Matrix(2, 2)  # Original 144 DPI for display
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Render HIGH QUALITY image for AI/OCR processing
        pix = page.get_pixmap(
            matrix=mat, 
            alpha=False,
            annots=True,
            clip=None
        )
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to grayscale if requested
        if color_mode == 'L':
            img = img.convert('L')
        
        # Apply enhancements to high-res image
        if enhance_quality:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            if sharpen:
                img = img.filter(ImageFilter.SHARPEN)
        
        images.append(img)
        
        # Create DISPLAY image at original resolution for UI
        display_pix = page.get_pixmap(matrix=display_mat, alpha=False)
        display_img = Image.frombytes("RGB", [display_pix.width, display_pix.height], display_pix.samples)
        
        if color_mode == 'L':
            display_img = display_img.convert('L')
        
        display_images.append(display_img)
    
    doc.close()
    return images, display_images

# ============= MODEL LOADER =============
@st.cache_resource
def load_vision_model(model_name, device="cuda"):
    """Universal loader for vision-language models with device selection"""
    try:
        progress = st.progress(0, text="Initializing...")
        
        is_cached = any(m['name'] == model_name and m['compatible'] for m in get_cached_models())
        
        if is_cached:
            st.info(f"📦 Loading {model_name} from cache...")
        else:
            st.info(f"📥 Downloading {model_name} (~{get_model_size(model_name)})... First run only!")
        
        # Clear memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        progress.progress(20, text="Loading processor...")
        
        # Determine dtype and device_map based on device
        if device == "cuda":
            dtype = torch.float16
            device_map = "auto"
            st.info("🚀 Loading model to GPU...")
        else:
            dtype = torch.float32
            device_map = None
            st.warning("🐌 Loading model to CPU (this may take a while)...")
        
        progress.progress(40, text=f"Loading model to {device.upper()}...")
        
        # Different loading strategies for different models
        if "qwen2-vl" in model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_name)
            
            if device == "cuda":
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            
        elif "llava" in model_name.lower():
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained(model_name)
            
            if device == "cuda":
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            
        elif "paligemma" in model_name.lower():
            from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
            processor = PaliGemmaProcessor.from_pretrained(model_name)
            
            if device == "cuda":
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            
        elif "blip2" in model_name.lower():
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(model_name)
            
            if device == "cuda":
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
            else:
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            
        elif "florence" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map=device_map,
                    attn_implementation="eager"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                )
                model = model.to(device)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        progress.progress(100, text="Complete!")
        st.success(f"✅ Model loaded successfully on {device.upper()}!")
        
        # Show memory usage
        if device == "cuda":
            vram = get_vram_info()
            ram_info = get_ram_info()
            st.info(f"📊 VRAM: {vram['vram_allocated_gb']:.2f}GB | RAM: {ram_info['ram_used_gb']:.1f}GB")
        else:
            ram_info = get_ram_info()
            st.info(f"📊 RAM: {ram_info['ram_used_gb']:.1f}GB ({ram_info['ram_percent']:.1f}%)")
        
        return processor, model, model_name
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        if "out of memory" in str(e).lower():
            if device == "cuda":
                st.warning("💡 **Out of VRAM! Try:**")
                st.code("""
1. Switch to CPU mode (slower but works)
2. Close other GPU applications
3. Try a smaller model (PaliGemma or Florence-2)
4. Clear GPU cache (button in sidebar)
                """)
            else:
                st.warning("💡 **Out of RAM! Try:**")
                st.code("""
1. Close other applications
2. Try a smaller model (Florence-2)
3. Increase virtual memory (Windows: System → Advanced)
                """)
        return None, None, None

# ============= MAIN UI =============
st.markdown("### 📄 Upload PDF Drawing")

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file:
    # ============= ADVANCED IMAGE QUALITY CONTROLS =============
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Image Quality Settings")
    
    # DPI Selection
    dpi_presets = {
        "Low (72 DPI) - Fast": 72,
        "Standard (150 DPI)": 150,
        "High (300 DPI) ⭐ Recommended": 300,
        "Very High (600 DPI)": 600,
        "Ultra (1200 DPI) - Slow": 1200,
        "Custom": 0
    }
    
    selected_dpi_preset = st.sidebar.selectbox(
        "Analysis Quality:",
        list(dpi_presets.keys()),
        index=2,  # Default to High (300 DPI)
        help="Quality used for AI analysis and OCR (not display)"
    )
    
    if dpi_presets[selected_dpi_preset] == 0:
        # Custom DPI slider
        selected_dpi = st.sidebar.slider(
            "Custom DPI:",
            min_value=72,
            max_value=1200,
            value=300,
            step=50,
            help="72-1200 DPI range"
        )
    else:
        selected_dpi = dpi_presets[selected_dpi_preset]
    
    # Color mode
    color_mode = st.sidebar.radio(
        "Color Mode:",
        ["RGB (Color)", "Grayscale"],
        index=0,
        help="Grayscale can improve OCR accuracy for text-heavy documents"
    )
    
    color_mode_value = 'L' if color_mode == "Grayscale" else 'RGB'
    
    # Enhancement options
    with st.sidebar.expander("🎨 Advanced Enhancements", expanded=False):
        enhance_quality = st.checkbox("Enhance Contrast", value=True, help="Improves text clarity")
        sharpen_image = st.checkbox("Sharpen Image", value=True, help="Makes text crisper")
        
        st.caption("⚠️ Enhancements may slightly increase processing time")
    
    st.sidebar.caption(f"**Display:** Original size (144 DPI)")
    st.sidebar.caption(f"**Analysis:** {selected_dpi} DPI ({color_mode})")
    
    if selected_dpi >= 600:
        st.sidebar.warning("⚠️ High DPI may be slow for multi-page PDFs")
    
    # Convert with selected settings
    with st.spinner(f"Loading PDF..."):
        images, display_images = pdf_to_images(
            uploaded_file, 
            dpi=selected_dpi,
            color_mode=color_mode_value,
            enhance_quality=enhance_quality,
            sharpen=sharpen_image
        )
    
    st.success(f"✅ Loaded {len(images)} page(s)")
    
    # Page selector
    page_num = st.slider("Select Page", 0, len(images)-1, 0)
    img = images[page_num]  # High-res for AI
    display_img = display_images[page_num]  # Original size for display
    
    # Show image info
    col1, col2 = st.columns([1, 3])
    with col1:
        st.caption(f"**Display:** {display_img.width} × {display_img.height} px")
        st.caption(f"**Analysis:** {img.width} × {img.height} px ({selected_dpi} DPI)")
    
    # Display at ORIGINAL SIZE
    st.image(display_img, caption=f"Page {page_num + 1} (Original Display Size)", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Analysis")
    
    # Model selection
    model_option = st.selectbox(
        "Select Vision Model:",
        [
            "Qwen/Qwen2-VL-7B-Instruct",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "google/paligemma-3b-mix-448",
            "Salesforce/blip2-flan-t5-xl",
            "microsoft/Florence-2-large"
        ],
        help="Qwen2-VL recommended for technical diagrams"
    )
    
    # Show model info
    with st.expander("ℹ️ Model Information"):
        model_info = {
            "Qwen/Qwen2-VL-7B-Instruct": {
                "vram": "8-9 GB",
                "ram": "16 GB",
                "speed": "Medium",
                "quality": "⭐⭐⭐⭐⭐",
                "best_for": "Technical diagrams, detailed OCR, equipment lists"
            },
            "llava-hf/llava-v1.6-mistral-7b-hf": {
                "vram": "8-9 GB",
                "ram": "16 GB",
                "speed": "Medium",
                "quality": "⭐⭐⭐⭐⭐",
                "best_for": "Complex instructions, detailed descriptions"
            },
            "google/paligemma-3b-mix-448": {
                "vram": "4-5 GB",
                "ram": "8 GB",
                "speed": "Fast",
                "quality": "⭐⭐⭐⭐",
                "best_for": "Balanced performance, limited VRAM"
            },
            "Salesforce/blip2-flan-t5-xl": {
                "vram": "7-8 GB",
                "ram": "16 GB",
                "speed": "Fast",
                "quality": "⭐⭐⭐",
                "best_for": "General descriptions, question answering"
            },
            "microsoft/Florence-2-large": {
                "vram": "2-3 GB",
                "ram": "8 GB",
                "speed": "Very Fast",
                "quality": "⭐⭐⭐⭐⭐",
                "best_for": "Pure OCR, text extraction, object detection"
            }
        }
        
        info = model_info.get(model_option, {})
        if device == "cpu":
            st.write(f"**RAM Required:** {info.get('ram', 'Unknown')}")
        else:
            st.write(f"**VRAM Required:** {info.get('vram', 'Unknown')}")
        st.write(f"**Speed:** {info.get('speed', 'Unknown')}")
        st.write(f"**Quality:** {info.get('quality', 'Unknown')}")
        st.write(f"**Best For:** {info.get('best_for', 'General use')}")
    
    # OCR toggle
    use_ocr_only = st.checkbox("🔤 Use OCR text extraction instead of AI", 
                               help="Fast text extraction using Tesseract (requires installation)")
    
    if not use_ocr_only:
        # Prompt templates
        prompt_templates = {
            "Detailed Inventory": "Analyze this data center rack diagram in detail. List each piece of equipment from top to bottom, including device names, model numbers, rack unit positions, labels, and any visible text or specifications.",
            "Quick List": "List all equipment visible in this rack diagram from top to bottom.",
            "With Cables": "Describe the equipment in this rack and any visible cable connections or network topology.",
            "OCR Focus": "Extract and list all visible text, labels, model numbers, and identifiers from this diagram.",
            "Custom": ""
        }
        
        template_choice = st.selectbox("Prompt Template:", list(prompt_templates.keys()))
        
        if template_choice == "Custom":
            prompt = st.text_area("Enter your custom prompt:", 
                                 "What equipment do you see in this rack?",
                                 height=100)
        else:
            prompt = st.text_area("Prompt (editable):", 
                                 prompt_templates[template_choice],
                                 height=150)
        
        # Analyze button
        if st.button("🔍 Analyze with AI", type="primary"):
            
            processor, model, model_type = load_vision_model(model_option, device)
            
            if processor and model:
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔍 Preparing image...")
                    progress_bar.progress(0.1)
                    
                    ram_before = get_ram_info()
                    if device == "cuda":
                        vram_before = get_vram_info()
                    
                    # Use HIGH-RES image for AI analysis
                    analysis_img = img
                    
                    # Model-specific inference with progress
                    if "qwen2-vl" in model_type.lower():
                        from qwen_vl_utils import process_vision_info
                        
                        status_text.text("📝 Processing prompt...")
                        progress_bar.progress(0.2)
                        
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": analysis_img},
                                {"type": "text", "text": prompt}
                            ]
                        }]
                        
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        
                        status_text.text("🖼️ Encoding image...")
                        progress_bar.progress(0.3)
                        
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(device)
                        
                        status_text.text("🤖 Generating response...")
                        progress_bar.progress(0.4)
                        
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=False
                            )
                            progress_bar.progress(0.9)
                        
                        status_text.text("📄 Decoding output...")
                        
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] 
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                    
                    elif "llava" in model_type.lower():
                        status_text.text("📝 Processing prompt...")
                        progress_bar.progress(0.2)
                        
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image"},
                                ],
                            },
                        ]
                        
                        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        
                        status_text.text("🖼️ Encoding image...")
                        progress_bar.progress(0.3)
                        
                        inputs = processor(images=analysis_img, text=prompt_text, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        status_text.text("🤖 Generating response...")
                        progress_bar.progress(0.4)
                        
                        with torch.no_grad():
                            output = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=False
                            )
                            progress_bar.progress(0.9)
                        
                        status_text.text("📄 Decoding output...")
                        
                        output_text = processor.decode(output[0], skip_special_tokens=True)
                        
                        if "[/INST]" in output_text:
                            output_text = output_text.split("[/INST]")[-1].strip()
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                    
                    elif "paligemma" in model_type.lower():
                        status_text.text("📝 Processing prompt...")
                        progress_bar.progress(0.2)
                        
                        inputs = processor(images=analysis_img, text=prompt, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        status_text.text("🤖 Generating response...")
                        progress_bar.progress(0.4)
                        
                        with torch.no_grad():
                            output = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=False
                            )
                            progress_bar.progress(0.9)
                        
                        status_text.text("📄 Decoding output...")
                        
                        output_text = processor.decode(output[0], skip_special_tokens=True)
                        output_text = output_text.replace(prompt, "").strip()
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                    
                    elif "blip2" in model_type.lower():
                        status_text.text("📝 Processing prompt...")
                        progress_bar.progress(0.2)
                        
                        inputs = processor(images=analysis_img, text=prompt, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        status_text.text("🤖 Generating response...")
                        progress_bar.progress(0.4)
                        
                        with torch.no_grad():
                            output = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=False
                            )
                            progress_bar.progress(0.9)
                        
                        status_text.text("📄 Decoding output...")
                        
                        output_text = processor.decode(output[0], skip_special_tokens=True)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                    
                    elif "florence" in model_type.lower():
                        status_text.text("📝 Processing prompt...")
                        progress_bar.progress(0.2)
                        
                        task_prompt = "<MORE_DETAILED_CAPTION>"
                        inputs = processor(text=task_prompt, images=analysis_img, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        status_text.text("🤖 Generating response...")
                        progress_bar.progress(0.4)
                        
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                num_beams=3
                            )
                            progress_bar.progress(0.9)
                        
                        status_text.text("📄 Decoding output...")
                        
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(analysis_img.width, analysis_img.height))
                        output_text = parsed_answer.get("<MORE_DETAILED_CAPTION>", "No description generated")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                    
                    else:
                        output_text = "Unknown model type"
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown("### 📝 Analysis Results")
                    st.markdown(output_text)
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📥 Download Results",
                        output_text,
                        f"analysis_{timestamp}.txt",
                        mime="text/plain"
                    )
                    
                    # Show resource usage
                    ram_after = get_ram_info()
                    if device == "cuda":
                        vram_after = get_vram_info()
                        st.info(f"⚡ VRAM: {vram_after['vram_allocated_gb']:.2f}GB | RAM: {ram_after['ram_used_gb']:.1f}GB")
                    else:
                        st.info(f"⚡ RAM: {ram_after['ram_used_gb']:.1f}GB ({ram_after['ram_percent']:.1f}%)")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error(f"❌ Error during inference: {str(e)}")
                    
                    if "out of memory" in str(e).lower():
                        if device == "cuda":
                            st.warning("💡 Try: Switch to CPU mode, close other apps, or use a smaller model")
                        else:
                            st.warning("💡 Try: Close other apps, use a smaller model, or increase virtual memory")
    
    else:
        # OCR Mode
        if st.button("🔍 Extract Text with OCR", type="primary"):
            try:
                import pytesseract
                from PIL import ImageEnhance, ImageFilter
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Auto-configure Tesseract path for Windows
                if platform.system() == 'Windows':
                    status_text.text("🔍 Looking for Tesseract...")
                    progress_bar.progress(0.1)
                    
                    possible_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    ]
                    
                    tesseract_found = False
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            tesseract_found = True
                            status_text.text(f"✅ Found Tesseract at: {path}")
                            progress_bar.progress(0.2)
                            break
                    
                    if not tesseract_found:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ **Tesseract not found!**")
                        st.markdown("""
                        **Windows Installation:**
                        
                        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
                        2. Run installer (use default path: `C:\\Program Files\\Tesseract-OCR`)
                        3. Restart this app
                        
                        **Or install manually to:** `C:\\Program Files\\Tesseract-OCR`
                        """)
                        st.stop()
                
                status_text.text("🖼️ Using high-res image for OCR...")
                progress_bar.progress(0.3)
                
                # Use HIGH-RES image for OCR
                ocr_img = img
                
                # Convert to grayscale if not already
                if ocr_img.mode != 'L':
                    ocr_img = ocr_img.convert('L')
                
                status_text.text("🎨 Enhancing for OCR...")
                progress_bar.progress(0.4)
                
                # Additional enhancement for OCR
                enhancer = ImageEnhance.Contrast(ocr_img)
                ocr_img = enhancer.enhance(1.5)
                
                status_text.text("✨ Sharpening...")
                progress_bar.progress(0.5)
                
                ocr_img = ocr_img.filter(ImageFilter.SHARPEN)
                
                status_text.text("📝 Extracting text (this may take 30-60 seconds)...")
                progress_bar.progress(0.6)
                
                # Extract text with optimized config for technical diagrams
                text = pytesseract.image_to_string(
                    ocr_img, 
                    config='--psm 6 --oem 3'
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if not text.strip():
                    st.warning("⚠️ No text detected. Try:")
                    st.info("- Higher DPI setting (600 or 1200)\n- Grayscale color mode\n- AI models instead")
                else:
                    st.markdown("### 📝 Extracted Text")
                    
                    # Show character and word count
                    char_count = len(text)
                    word_count = len(text.split())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Characters", f"{char_count:,}")
                    with col2:
                        st.metric("Words", f"{word_count:,}")
                    
                    st.text_area("", text, height=400)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📥 Download Text", 
                        text, 
                        f"ocr_{timestamp}.txt",
                        mime="text/plain"
                    )
                    
                    ram_after = get_ram_info()
                    st.info(f"⚡ RAM: {ram_after['ram_used_gb']:.1f}GB ({ram_after['ram_percent']:.1f}%)")
                    
            except ImportError:
                st.error("❌ **pytesseract not installed**")
                st.code("pip install pytesseract")
                
            except Exception as e:
                if "TesseractNotFoundError" in str(type(e).__name__):
                    st.error("❌ **Tesseract OCR not installed**")
                    
                    if platform.system() == 'Windows':
                        st.markdown("""
                        ### Windows Installation:
                        
                        1. **Download:** https://github.com/UB-Mannheim/tesseract/wiki
                        2. **Install** to default location: `C:\\Program Files\\Tesseract-OCR`
                        3. **Restart** this app
                        
                        **Current search paths:**
                        - `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
                        - `C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe`
                        """)
                    elif platform.system() == 'Linux':
                        st.code("sudo apt-get install tesseract-ocr")
                    elif platform.system() == 'Darwin':
                        st.code("brew install tesseract")
                else:
                    st.error(f"❌ Error: {e}")
                    st.info("💡 Try using AI models instead (uncheck OCR option)")

else:
    st.info("👆 Upload a PDF to get started")
    
    st.markdown("---")
    st.markdown("### 📖 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Upload**
        - Click 'Browse files' above
        - Select your data center rack diagram PDF
        - PDF displays at original size
        
        **Step 2: Choose Quality**
        - Adjust "Analysis Quality" in sidebar
        - **300 DPI** - Recommended ⭐
        - **600+ DPI** - For tiny text
        - Display stays at original size
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Analyze**
        - Select AI model or OCR mode
        - Choose prompt template or write custom
        - Analysis uses high-res, display stays sharp
        
        **Tips:**
        - Display: Always original size (fast)
        - Analysis: Uses your selected DPI
        - Higher DPI = better AI accuracy
        - First model load downloads ~5-15 GB
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 How It Works")
    
    st.info("""
    **Dual Resolution System:**
    
    📺 **Display Image:** Original size (144 DPI) - Fast, sharp preview in browser
    
    🤖 **Analysis Image:** Your selected DPI (300-1200) - High quality for AI/OCR
    
    This gives you fast loading and sharp display while maintaining high quality for analysis!
    """)
    
    if cuda_info['pytorch_cuda']:
        vram = get_vram_info()
        vram_gb = vram['vram_total_gb'] if vram else 0
        
        st.markdown("---")
        
        if vram_gb >= 10:
            st.success("🚀 **Your GPU is perfect for all models!**")
            st.write("Recommended: Qwen2-VL-7B or LLaVA-v1.6 for best quality")
        elif vram_gb >= 6:
            st.info("✅ **Your GPU can run most models**")
            st.write("Recommended: PaliGemma-3B or BLIP2 for balanced performance")
        elif vram_gb >= 4:
            st.warning("⚠️ **Your GPU has limited VRAM**")
            st.write("Recommended: Florence-2 or switch to CPU mode")
        else:
            st.error("❌ **Very limited VRAM**")
            st.write("Recommended: Use CPU mode with Florence-2")
    else:
        st.info("💻 **CPU Mode Active**")
        st.write("Recommended: Florence-2 (fastest) or PaliGemma (better quality)")
        st.write("⚠️ Expect 2-5 minute processing time per image")
