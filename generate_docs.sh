#!/bin/bash

# llama.cpp API Documentation Generator
# Generates Doxygen documentation for GGML_API, GGML_BACKEND_API, LLAMA_API, and MTMD_API functions

echo "=== llama.cpp API Documentation Generator ==="

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "❌ Error: Doxygen is not installed."
    echo "Please install Doxygen first:"
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  CentOS/RHEL: sudo yum install doxygen"
    exit 1
fi

# Check if Graphviz (dot) is installed for graphs
if ! command -v dot &> /dev/null; then
    echo "⚠️  Warning: Graphviz (dot) is not installed."
    echo "  Graphs will not be generated. To install:"
    echo "  macOS: brew install graphviz"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  CentOS/RHEL: sudo yum install graphviz"
    echo ""
fi

# Create output directory
echo "📁 Creating output directory..."
mkdir -p docs

# Get API function counts
echo "📊 Analyzing API functions..."

LLAMA_API_COUNT=$(grep -r "LLAMA_API" include/ --include="*.h" | grep -v "#define" | wc -l)
GGML_API_COUNT=$(grep -r "GGML_API" ggml/include/ --include="*.h" | grep -v "#define" | wc -l)
GGML_BACKEND_API_COUNT=$(grep -r "GGML_BACKEND_API" ggml/include/ --include="*.h" | grep -v "#define" | wc -l)
MTMD_API_COUNT=$(grep -r "MTMD_API" tools/mtmd/ --include="*.h" | grep -v "#define" | wc -l)

echo "  - LLAMA_API functions: $LLAMA_API_COUNT"
echo "  - GGML_API functions: $GGML_API_COUNT"
echo "  - GGML_BACKEND_API functions: $GGML_BACKEND_API_COUNT"
echo "  - MTMD_API functions: $MTMD_API_COUNT"

TOTAL_API_COUNT=$((LLAMA_API_COUNT + GGML_API_COUNT + GGML_BACKEND_API_COUNT + MTMD_API_COUNT))
echo "  - Total API functions: $TOTAL_API_COUNT"

# Generate documentation
echo ""
echo "🔧 Generating Doxygen documentation..."
doxygen Doxyfile

# Check if documentation was generated successfully
if [ -d "docs/html" ] && [ -f "docs/html/index.html" ]; then
    echo ""
    echo "✅ Documentation generated successfully!"
    echo ""
    echo "📖 Documentation location:"
    echo "   HTML: $(pwd)/docs/html/index.html"
    echo ""
    echo "🌐 To view the documentation:"
    echo "   open docs/html/index.html"
    echo "   # or"
    echo "   python3 -m http.server 8000 --directory docs/html"
    echo "   # then open http://localhost:8000"
    echo ""
    echo "📋 API Coverage:"
    echo "   - LLAMA API: include/llama.h"
    echo "   - GGML API: ggml/include/ggml.h"
    echo "   - GGML Backend API: ggml/include/ggml-backend.h"
    echo "   - MTMD API: tools/mtmd/mtmd.h"
    echo ""
    
    # Calculate file sizes
    HTML_SIZE=$(du -sh docs/html 2>/dev/null | cut -f1)
    echo "📏 Documentation size: $HTML_SIZE"
    
else
    echo ""
    echo "❌ Documentation generation failed!"
    echo "Please check the Doxygen warnings above."
    exit 1
fi
