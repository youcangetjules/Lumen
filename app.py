import streamlit as st
import fitz
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
import gc
from pathlib import Path
import shutil
from datetime import datetime
import psutil

st.set_page_config(page_title="Data Center Inventory Extractor", layout="wide")

st.title("Data Center Inventory Extractor")

APP_VERSION = "3.0"
COMPATIBLE_MODELS = {
    "Salesforce/blip2-opt-2.7b": "3.0",
    "Salesforce/blip2-flan-t5-xl": "3.0",
    "microsoft/Florence-2-large": "3.0",
    "Qwen/Qwen2-VL-7B-Instruct": "3.0",
    "llava-hf/llava-v1.6-mistral-7b-hf": "3.0",
    "google/paligemma-3b-mix-448": "3.0"
}

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

# System info sidebar
st.sidebar.markdown("### 💻 System Info")
st.sidebar.write(f"**GPU:** RTX 3080 (10GB)")

ram_info = get_ram_info()
st.sidebar.write(f"**RAM:** {ram_info['ram_available_gb']:.1f}GB / {ram_info['ram_total_gb']:.1f}GB free")
st.sidebar.write(f"**RAM Usage:** {ram_info['ram_percent']:.1f}%")
st.sidebar.write(f"**Virtual RAM:** {ram_info['swap_used_gb']:.1f}GB / {ram_info['swap_total_gb']:.1f}GB used")

if torch.cuda.is_available():
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
    vram_free = vram_total - vram_allocated
    st.sidebar.write(f"**VRAM Free:** {vram_free:.1f}GB / {vram_total:.1f}GB")

st.sidebar.markdown("---")

st.sidebar.markdown("### 💾 Model Cache Manager")
st.sidebar.caption(f"App Version: {APP_VERSION}")

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

def get_cached_models():
    """Get all cached models with details"""
    if not cache_dir.exists():
        return []
    
    cached_models = []
    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
    
    for model_dir in model_dirs:
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = list(snapshots_dir.iterdir())
            if snapshot_dirs:
                last_modified = max(d.stat().st_mtime for d in snapshot_dirs)
                last_modified_date = datetime.fromtimestamp(last_modified)
            else:
                last_modified_date = None
        else:
            last_modified_date = None
        
        is_compatible = model_name in COMPATIBLE_MODELS
        required_version = COMPATIBLE_MODELS.get(model_name, "Unknown")
        
        cached_models.append({
            'name': model_name,
            'short_name': model_name.split('/')[-1],
            'size_gb': size / 1024**3,
            'path': model_dir,
            'last_modified': last_modified_date,
            'compatible': is_compatible,
            'required_version': required_version
        })
    
    return cached_models

cached_models = get_cached_models()

if cached_models:
    total_size = sum(m['size_gb'] for m in cached_models)
    st.sidebar.write(f"**Total Cache:** {total_size:.2f} GB")
    st.sidebar.write(f"**Models Cached:** {len(cached_models)}")
    
    with st.sidebar.expander("📦 Cached Models Details", expanded=False):
        for model in cached_models:
            if model['compatible']:
                st.success(f"✅ **{model['short_name']}**")
                st.caption(f"   Size: {model['size_gb']:.2f} GB")
                if model['last_modified']:
                    st.caption(f"   Updated: {model['last_modified'].strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning(f"⚠️ **{model['short_name']}**")
                st.caption(f"   Size: {model['size_gb']:.2f} GB")
                st.caption(f"   ⚠️ STALE - Not compatible with v{APP_VERSION}")
        
        st.markdown("---")
        st.markdown("**Remove Individual Models:**")
        for model in cached_models:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(model['short_name'])
            with col2:
                if st.button("🗑️", key=f"del_{model['name']}"):
                    try:
                        shutil.rmtree(model['path'])
                        st.success(f"Deleted {model['short_name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            try:
                shutil.rmtree(cache_dir)
                st.success("All models cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        stale_count = sum(1 for m in cached_models if not m['compatible'])
        if stale_count > 0:
            if st.button(f"🧹 Remove Stale ({stale_count})", use_container_width=True):
                try:
                    for model in cached_models:
                        if not model['compatible']:
                            shutil.rmtree(model['path'])
                    st.success(f"Removed {stale_count} stale model(s)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.button("✨ All Fresh", use_container_width=True, disabled=True)

else:
    st.sidebar.info("No models cached yet")

st.sidebar.markdown("---")

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

def get_model_info(model_name):
    """Get model capabilities"""
    info = {
        "Salesforce/blip2-opt-2.7b": "Fast, general descriptions",
        "Salesforce/blip2-flan-t5-xl": "Better reasoning, Q&A",
        "microsoft/Florence-2-large": "OCR, technical diagrams",
        "Qwen/Qwen2-VL-7B-Instruct": "⭐ BEST for technical diagrams + OCR",
        "llava-hf/llava-v1.6-mistral-7b-hf": "⭐ Excellent instruction following",
        "google/paligemma-3b-mix-448": "Good balance, lighter weight"
    }
    return info.get(model_name, "")

model_option = st.sidebar.selectbox(
    "Select Model",
    [
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "google/paligemma-3b-mix-448",
        "Salesforce/blip2-flan-t5-xl",
        "microsoft/Florence-2-large",
        "Salesforce/blip2-opt-2.7b"
    ],
    index=0,
    help="Qwen2-VL and LLaVA are best for technical diagrams"
)

st.sidebar.caption(get_model_info(model_option))

selected_cached = any(m['name'] == model_option and m['compatible'] for m in cached_models)
if selected_cached:
    st.sidebar.success(f"✅ {model_option.split('/')[-1]} is cached")
else:
    st.sidebar.info(f"📥 Will download on first use (~{get_model_size(model_option)})")

@st.cache_resource
def load_vision_model(model_name):
    """Universal loader for vision-language models"""
    try:
        progress = st.progress(0, text="Initializing...")
        
        is_cached = any(m['name'] == model_name and m['compatible'] for m in get_cached_models())
        
        if is_cached:
            st.info(f"📦 Loading {model_name} from cache...")
        else:
            st.info(f"📥 Downloading {model_name} (~{get_model_size(model_name)})... First run only!")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        progress.progress(20, text="Loading processor...")
        
        # Different loading strategies for different models
        if "qwen2-vl" in model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_name)
            
            progress.progress(40, text="Loading model to GPU...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        elif "llava" in model_name.lower():
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained(model_name)
            
            progress.progress(40, text="Loading model to GPU...")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        elif "paligemma" in model_name.lower():
            from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
            processor = PaliGemmaProcessor.from_pretrained(model_name)
            
            progress.progress(40, text="Loading model to GPU...")
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        elif "blip2" in model_name.lower():
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(model_name)
            
            progress.progress(40, text="Loading model to GPU...")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        elif "florence" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            progress.progress(40, text="Loading model to GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="eager"
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        progress.progress(100, text="Complete!")
        st.success(f"✅ Model loaded successfully!")
        
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        ram_info = get_ram_info()
        st.info(f"📊 VRAM: {vram_used:.2f}GB | RAM: {ram_info['ram_used_gb']:.1f}GB")
        
        return processor, model, model_name
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        if "out of memory" in str(e).lower():
            st.warning("💡 **Out of VRAM! Try:**")
            st.code("""
1. Close other GPU applications
2. Try a smaller model (PaliGemma or Florence-2)
3. Clear GPU cache (button in sidebar)
            """)
        return None, None, None

uploaded_file = st.file_uploader("Upload PDF drawing", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_size_mb = len(pdf_bytes) / 1024**2
    
    ram_info = get_ram_info()
    estimated_ram_needed = pdf_size_mb * 3
    
    st.info(f"📄 PDF Size: {pdf_size_mb:.2f} MB | Estimated RAM needed: ~{estimated_ram_needed:.0f} MB")
    
    if ram_info['ram_available_gb'] * 1024 < estimated_ram_needed:
        st.warning(f"⚠️ Low RAM! Available: {ram_info['ram_available_gb']:.1f}GB. Close other applications.")
    
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    num_pages = len(pdf_document)
    page_num = st.slider("Select page", 1, num_pages, 1) - 1
    
    page = pdf_document[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    ram_after = get_ram_info()
    st.caption(f"Page loaded | RAM: {ram_after['ram_used_gb']:.1f}GB used ({ram_after['ram_percent']:.1f}%)")
    
    st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
    
    st.subheader("🤖 AI Analysis")
    
    # OCR-only option
    use_ocr_only = st.checkbox("🔤 Use OCR text extraction instead of AI vision", 
                               help="Extract readable text directly - faster and more accurate for text-heavy diagrams")
    
    if use_ocr_only:
        if st.button("🔍 Extract Text with OCR", type="primary"):
            try:
                import pytesseract
                from PIL import ImageEnhance, ImageFilter
                
                with st.spinner("Extracting text..."):
                    gray = img.convert('L')
                    enhancer = ImageEnhance.Contrast(gray)
                    enhanced = enhancer.enhance(2.0)
                    sharpened = enhanced.filter(ImageFilter.SHARPEN)
                    
                    text = pytesseract.image_to_string(sharpened, config='--psm 6')
                    
                    st.markdown("### 📝 Extracted Text")
                    st.text_area("", text, height=400)
                    
                    st.download_button(
                        "📥 Download Text", 
                        text, 
                        f"ocr_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                    
            except ImportError:
                st.error("**Install Tesseract OCR:**")
                st.code("pip install pytesseract")
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        prompt_template = st.selectbox(
            "Select Prompt Template",
            [
                "Detailed Inventory",
                "Simple Description",
                "List All Text",
                "Count Equipment",
                "Custom"
            ]
        )

        if prompt_template == "Detailed Inventory":
            default_prompt = "Analyze this data center rack diagram in detail. List each piece of equipment from top to bottom, including device names, model numbers, rack unit positions, labels, and any visible text or specifications."

        elif prompt_template == "Simple Description":
            default_prompt = "Describe all equipment visible in this rack diagram. What devices do you see?"

        elif prompt_template == "List All Text":
            default_prompt = "Read and transcribe all text visible in this image, including labels, model numbers, port numbers, and any other readable text."

        elif prompt_template == "Count Equipment":
            default_prompt = "Count and list each piece of equipment in this rack, numbering them from top to bottom."

        else:
            default_prompt = "Analyze this technical diagram and provide a detailed inventory of all visible equipment and text."

        prompt = st.text_area(
            "Prompt:",
            value=default_prompt,
            height=120
        )
        
        if st.button("🔍 Analyze with AI", type="primary"):
            
            processor, model, model_type = load_vision_model(model_option)
            
            if processor and model:
                with st.spinner("🔍 Analyzing image..."):
                    try:
                        # Different inference for different models
                        if "qwen2-vl" in model_type.lower():
                            # Qwen2-VL specific
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": img},
                                        {"type": "text", "text": prompt}
                                    ]
                                }
                            ]
                            
                            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                do_sample=False
                            )
                            
                            result = processor.batch_decode(
                                generated_ids[:, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )[0]
                            
                        elif "llava" in model_type.lower():
                            # LLaVA specific
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image"},
                                        {"type": "text", "text": prompt}
                                    ]
                                }
                            ]
                            
                            prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                            inputs = processor(images=img, text=prompt_text, return_tensors="pt")
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                do_sample=False
                            )
                            
                            result = processor.decode(generated_ids[0], skip_special_tokens=True)
                            if "ASSISTANT:" in result:
                                result = result.split("ASSISTANT:")[-1].strip()
                            
                        elif "paligemma" in model_type.lower():
                            # PaliGemma specific
                            inputs = processor(text=prompt, images=img, return_tensors="pt")
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                do_sample=False
                            )
                            
                            result = processor.decode(generated_ids[0], skip_special_tokens=True)
                            
                        elif "florence" in model_type.lower():
                            # Florence-2 specific
                            task_prompt = "<DETAILED_CAPTION>"
                            inputs = processor(text=task_prompt, images=img, return_tensors="pt")
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            generated_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                num_beams=3
                            )
                            
                            result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                            result = processor.post_process_generation(
                                result, 
                                task=task_prompt, 
                                image_size=(img.width, img.height)
                            )
                            
                            st.markdown("### 📋 Analysis")
                            import json
                            st.json(result)
                            
                            vram_used = torch.cuda.memory_allocated(0) / 1024**3
                            ram_after = get_ram_info()
                            st.info(f"⚡ VRAM: {vram_used:.2f}GB | RAM: {ram_after['ram_used_gb']:.1f}GB")
                            
                            st.download_button(
                                "📥 Download",
                                json.dumps(result, indent=2),
                                f"analysis_{uploaded_file.name}.json",
                                mime="application/json"
                            )
                            
                            pdf_document.close()
                            st.stop()
                            
                        else:
                            # BLIP-2 fallback
                            inputs = processor(images=img, text=prompt, return_tensors="pt")
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                            
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                num_beams=5,
                                no_repeat_ngram_size=3
                            )
                            
                            result = processor.decode(generated_ids[0], skip_special_tokens=True)
                        
                        st.markdown("### 📋 Analysis Result")
                        st.write(result)
                        
                        with st.expander("📊 Details"):
                            st.write(f"**Length:** {len(result)} characters")
                            st.write(f"**Words:** {len(result.split())}")
                        
                        vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        ram_after = get_ram_info()
                        st.info(f"⚡ VRAM: {vram_used:.2f}GB | RAM: {ram_after['ram_used_gb']:.1f}GB")
                        
                        st.download_button(
                            "📥 Download Analysis",
                            result,
                            f"analysis_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
                        if "out of memory" in str(e).lower():
                            st.warning("Try a smaller model or clear GPU cache")
    
    pdf_document.close()

else:
    st.info("👆 Upload a PDF to get started")
    
    st.markdown("""
    ### 🚀 Recommended Models for Technical Diagrams:
    
    | Model | Size | Best For | RTX 3080 |
    |-------|------|----------|----------|
    | **⭐ Qwen2-VL-7B** | 15 GB | Technical diagrams, OCR, detailed analysis | ✅ Fits |
    | **⭐ LLaVA-v1.6-Mistral** | 15 GB | Following complex instructions | ✅ Fits |
    | **PaliGemma-3B** | 6 GB | Good balance, lighter weight | ✅ Fits |
    | **BLIP-2 Flan-T5-XL** | 11 GB | General Q&A | ✅ Fits |
    | **Florence-2** | 1.5 GB | Fast OCR | ✅ Fits |
    
    ### 💡 Tips:
    - **Try Qwen2-VL first** - it's specifically designed for technical documents
    - LLaVA is excellent at following detailed instructions
    - Use OCR mode for pure text extraction
    - Close other GPU apps before loading large models
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"App Version: {APP_VERSION}")
if st.sidebar.button("🔄 Clear All Caches & Restart"):
    st.cache_resource.clear()
    st.cache_data.clear()
    torch.cuda.empty_cache()
    gc.collect()
    st.sidebar.success("Cleared! Reloading...")
    st.rerun()
