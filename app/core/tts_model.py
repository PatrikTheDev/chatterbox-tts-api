"""
TTS model initialization and management
"""

import os
import asyncio
from enum import Enum
from typing import Optional
from pathlib import Path
from chatterbox_vllm.tts import ChatterboxTTS
from app.config import Config, detect_device

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress
    
    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None
            
            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)
            
            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')
            
            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file
        
        _initialization_progress = "Loading TTS model (this may take a while)..."
        
        # Debug: Check current working directory and t3-model structure
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")
        t3_model_dir = Path("./t3-model")
        print(f"t3-model directory exists: {t3_model_dir.exists()}")
        if t3_model_dir.exists():
            print(f"t3-model contents: {list(t3_model_dir.iterdir())}")
        
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        def init_model_with_symlinks():
            """Initialize model and ensure proper symlinks for vLLM"""
            # Import here to avoid circular imports
            from huggingface_hub import hf_hub_download
            
            # Download model files if not cached
            repo_id = "ResembleAI/chatterbox"
            revision = "1b475dffa71fb191cb6d5901215eb6f55635a9b6"
            
            # Ensure all required files are downloaded
            for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
                local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)
            
            # Create symlinks for vLLM model directory
            cache_dir = Path(local_path).parent
            t3_model_dir = Path("./t3-model")
            
            # Symlink model weights
            model_safetensors_path = t3_model_dir / "model.safetensors"
            model_safetensors_path.unlink(missing_ok=True)
            model_safetensors_path.symlink_to(cache_dir / "t3_cfg.safetensors")
            
            # Symlink tokenizer (in case vLLM expects it in model dir)
            tokenizer_path = t3_model_dir / "tokenizer.json" 
            tokenizer_path.unlink(missing_ok=True)
            tokenizer_path.symlink_to(cache_dir / "tokenizer.json")
            
            print(f"✓ Created symlinks for vLLM model directory")
            
            # Debug: Check final t3-model directory structure
            print(f"Final t3-model directory contents:")
            for item in t3_model_dir.iterdir():
                if item.is_symlink():
                    target = item.readlink()
                    target_exists = target.exists()
                    print(f"  {item.name} -> {target} (exists: {target_exists})")
                else:
                    print(f"  {item.name} (regular file)")
            
            # Check if all required files exist
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]
            for filename in required_files:
                filepath = t3_model_dir / filename
                exists = filepath.exists()
                print(f"Required file {filename}: exists={exists}")
                if filepath.is_symlink():
                    target = filepath.readlink()
                    print(f"  -> symlink target: {target} (target exists: {target.exists()})")
            
            # Now initialize the model
            print("Attempting to initialize ChatterboxTTS...")
            
            # Test if we can import EnTokenizer first
            try:
                from chatterbox_vllm.models.t3.entokenizer import EnTokenizer
                print("✓ EnTokenizer can be imported")
                
                # Test creating an instance
                tokenizer_test = EnTokenizer.from_pretrained()
                print("✓ EnTokenizer can be instantiated")
            except Exception as tokenizer_error:
                print(f"✗ EnTokenizer issue: {tokenizer_error}")
            
            try:
                model = ChatterboxTTS.from_pretrained(
                    target_device=_device,
                    max_batch_size=10,
                    max_model_len=1000
                )
                print("✓ ChatterboxTTS initialized successfully")
                return model
            except Exception as e:
                print(f"✗ ChatterboxTTS initialization failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                
                # Additional debugging for common vLLM errors
                if "No such file or directory" in str(e):
                    print("This appears to be a missing file error. Checking common vLLM requirements...")
                    
                    # Check if vLLM can find the model directory
                    print(f"Model directory './t3-model' exists: {Path('./t3-model').exists()}")
                    
                    # Check for additional files vLLM might expect
                    possible_files = ["pytorch_model.bin", "model.bin", "generation_config.json", "tokenizer_config.json"]
                    for possible_file in possible_files:
                        filepath = t3_model_dir / possible_file
                        print(f"Optional file {possible_file}: exists={filepath.exists()}")
                
                raise e
        
        _model = await loop.run_in_executor(None, init_model_with_symlinks)
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value


def shutdown_model():
    """Shutdown the model and cleanup resources"""
    global _model, _initialization_state
    if _model is not None:
        try:
            _model.shutdown()
            print("✓ Model shutdown completed")
        except Exception as e:
            print(f"⚠️ Warning during model shutdown: {e}")
        finally:
            _model = None
            _initialization_state = InitializationState.NOT_STARTED.value