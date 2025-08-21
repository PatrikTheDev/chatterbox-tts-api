# T3 Model Directory

This directory contains the vLLM model configuration files required for chatterbox-vllm.

## Files:
- `config.json` - vLLM model configuration (static)
- `tokenizer.json` - Tokenizer configuration (will be symlinked during initialization)
- `model.safetensors` - Model weights (will be symlinked during initialization)

## Notes:
The `tokenizer.json` file here may be symlinked during runtime from the HuggingFace cache
after the chatterbox-vllm package downloads the model files. If this causes issues,
we can remove it and let the package handle tokenizer loading from its internal location.

The chatterbox-vllm package expects this directory structure for vLLM to properly load
the custom T3 model with the EnTokenizer.