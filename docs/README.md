# llama.cpp API Documentation

This directory contains comprehensive API documentation for llama.cpp generated using Doxygen.

## Overview

The documentation covers **942 API functions** across four main API categories:

- **LLAMA_API** (224 functions): Main llama.cpp API for model loading, inference, and text processing
- **GGML_API** (534 functions): Core tensor operations and machine learning primitives  
- **GGML_BACKEND_API** (132 functions): Backend management for GPU/CPU compute
- **MTMD_API** (52 functions): Multimodal (audio/vision) processing capabilities

## API Categories

### 📚 LLAMA API (`include/llama.h`)
The main C API providing:
- **Model Management**: Loading, saving, and freeing models
- **Context Operations**: Creating and managing inference contexts
- **Tokenization**: Converting between text and tokens
- **Inference**: Running model inference and sampling
- **Memory Management**: KV cache and state management
- **Adapters**: LoRA and control vector support

### 🔧 GGML API (`ggml/include/ggml.h`)
Core tensor library providing:
- **Tensor Operations**: Creation, manipulation, and arithmetic
- **Neural Network Layers**: Convolution, attention, normalization
- **Automatic Differentiation**: Gradient computation
- **Optimization**: Training and fine-tuning support
- **Memory Management**: Context allocation and tensor lifetimes

### ⚡ GGML Backend API (`ggml/include/ggml-backend.h`)
Backend abstraction layer providing:
- **Device Management**: CPU, CUDA, Metal, OpenCL backends
- **Memory Allocation**: Device-specific buffer management
- **Compute Scheduling**: Graph execution and optimization
- **Buffer Operations**: Data transfer between devices

### 🎭 MTMD API (`tools/mtmd/mtmd.h`)
Multimodal processing API providing:
- **Vision Processing**: Image encoding and understanding
- **Audio Processing**: Speech and audio analysis
- **Bitmap Operations**: Image manipulation utilities
- **Context Integration**: Multimodal context management

## Documentation Structure

```
docs/
├── html/
│   ├── index.html          # Main documentation entry point
│   ├── modules.html        # API groups overview
│   ├── files.html          # Source file browser
│   ├── annotated.html      # Data structures index
│   ├── functions.html      # Function index
│   └── search/             # Search functionality
├── Doxyfile               # Doxygen configuration
└── generate_docs.sh       # Documentation generator script
```

## Viewing the Documentation

### Option 1: Direct File Access
```bash
open docs/html/index.html
```

### Option 2: Local Web Server
```bash
# Start a local web server
python3 -m http.server 8000 --directory docs/html

# Open in browser
open http://localhost:8000
```

### Option 3: VS Code Live Server
1. Install the "Live Server" extension in VS Code
2. Right-click on `docs/html/index.html`
3. Select "Open with Live Server"

## Regenerating Documentation

To update the documentation after code changes:

```bash
./generate_docs.sh
```

### Requirements
- **Doxygen**: For documentation generation
  - macOS: `brew install doxygen`
  - Ubuntu/Debian: `sudo apt-get install doxygen`
  - CentOS/RHEL: `sudo yum install doxygen`

- **Graphviz** (optional): For dependency graphs and diagrams
  - macOS: `brew install graphviz` 
  - Ubuntu/Debian: `sudo apt-get install graphviz`
  - CentOS/RHEL: `sudo yum install graphviz`

## Navigation Tips

### Finding Functions
1. **By Category**: Use the "Modules" page to browse by API category
2. **By Name**: Use the "Functions" index for alphabetical listing
3. **Search**: Use the search box in the top-right corner
4. **File Browse**: Use "Files" to browse by header file

### Understanding Function Signatures
- **Return Types**: Clearly documented with descriptions
- **Parameters**: Input/output parameters with detailed explanations
- **Examples**: Usage examples where available
- **Related Functions**: Cross-references to related API calls

### Key Documentation Sections

#### For Developers
- **Function Reference**: Complete API documentation
- **Data Structures**: Detailed struct and enum documentation
- **Examples**: Usage patterns and code samples
- **Dependencies**: Include file requirements

#### For Integrators
- **Getting Started**: Basic API usage patterns
- **Error Handling**: Return codes and error management
- **Memory Management**: Allocation and cleanup patterns
- **Thread Safety**: Concurrency considerations

## API Stability

- **LLAMA_API**: Stable public API with deprecation warnings
- **GGML_API**: Core API, relatively stable
- **GGML_BACKEND_API**: Evolving, backend-specific details may change
- **MTMD_API**: Experimental, may change in future versions

## Contributing

When adding new API functions:

1. **Add Doxygen Comments**: Use `/** */` style with `@param`, `@return`, `@brief`
2. **Update Groups**: Add functions to appropriate `@defgroup` sections
3. **Regenerate Docs**: Run `./generate_docs.sh` to update documentation
4. **Review Output**: Check for warnings and missing documentation

### Documentation Standards

```c
/**
 * @brief Brief description of the function
 * @details Detailed description if needed
 * 
 * @param[in] input_param Description of input parameter
 * @param[out] output_param Description of output parameter
 * @param[in,out] inout_param Description of input/output parameter
 * 
 * @return Description of return value
 * @retval 0 Success
 * @retval -1 Error condition
 * 
 * @note Important notes about usage
 * @warning Warnings about potential issues
 * 
 * @see related_function()
 * @since Version when function was added
 */
LLAMA_API int example_function(const char* input_param, char* output_param);
```

## File Locations

- **Main API**: `include/llama.h`
- **GGML Core**: `ggml/include/ggml.h`
- **GGML Backend**: `ggml/include/ggml-backend.h` 
- **Multimodal**: `tools/mtmd/mtmd.h`
- **Documentation Config**: `Doxyfile`
- **Generator Script**: `generate_docs.sh`

## Documentation Size

Current documentation size: **~1.9MB** covering 942 API functions across 25 header files.

---

*Generated by Doxygen from llama.cpp source code*
