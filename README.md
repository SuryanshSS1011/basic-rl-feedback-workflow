# Secure Code Generation Workflow

A complete pipeline for **LLM-based code generation** with **security analysis** using CodeQL and **symbolic execution** with KLEE - all without requiring administrator privileges.

## 🚀 Features

- **LLM Code Generation**: DeepSeek/HuggingFace models with automatic caching and cleanup
- **CodeQL Security Analysis**: Static security analysis with GitHub's CodeQL
- **KLEE Symbolic Execution**: Comprehensive path exploration and test case generation
- **User-Space Installation**: No sudo/admin privileges required - username auto-detected
- **Unified Pipeline**: Single command runs complete generation → analysis workflow
- **Smart Code Cleaning**: Removes LLM artifacts, comments, and duplicate statements
- **Disk Quota Friendly**: Uses `/scratch` space and cleans up automatically

## 📋 Prerequisites

- **Python 3.11** (system-wide installation)
- **CUDA 12.4** (for GPU-accelerated LLM inference)
- **Git** and basic build tools (gcc, make)
- **Internet connection** (for downloading dependencies)

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd basic-rl-feedback-workflow
```

### Step 2: Run Complete Setup
**Note**: Your username is automatically detected - no manual configuration needed!
```bash
chmod +x prerequisites-setup.sh
./prerequisites-setup.sh
```

This will install and build:
- Python 3.11 virtual environment with PyTorch, Transformers, llama-cpp-python
- CodeQL static analysis engine
- CMake, Ninja, LLVM 14.0 (user-space)
- SQLite 3.43.2 (built from source)
- Z3 4.8.15 SMT solver (built from source)
- KLEE symbolic execution engine (built from source)

**⏱️ Installation time: ~30-45 minutes** (depending on your system)

## 🎯 Usage

### Generate and Analyze Code
```bash
# Run the complete pipeline (LLM generation + analysis)
./run_pipeline.sh

# OR run analysis only on existing code
./analyze_only.sh
```

**Complete Pipeline** (`run_pipeline.sh`):
1. **Generate C code** using LLM (DeepSeek model)
2. **Clean and process** code (remove markdown, duplicates)
3. **Run CodeQL security analysis** to detect vulnerabilities
4. **Generate LLVM bitcode** for symbolic execution
5. **Execute KLEE analysis** with 30-second timeout, generating test cases

### Pipeline Options

#### Complete Pipeline (Recommended)
```bash
./run_pipeline.sh
```
- **Generates** new C code with LLM
- **Analyzes** with CodeQL and KLEE
- **Everything** in one command

#### Analysis Only
```bash
./analyze_only.sh
```
- **Re-analyzes** existing generated code
- **Faster** - skips LLM generation
- **Useful** for testing different analysis parameters

#### Individual Components

**Generate Code Only:**
```bash
python run_llm.py
```
- Output: `generated_code/generated_code.c`

**View Results:**
```bash
# View generated code
cat generated_code/generated_code.c

# View KLEE test cases
ls -la klee_output/
/scratch/$(whoami)/klee/build/bin/ktest-tool klee_output/test*.ktest

# View analysis statistics
cat klee_output/info
```

## 📁 Project Structure

```
basic-rl-feedback-workflow/
├── prerequisites-setup.sh    # Complete setup script (auto-detects username)
├── run_pipeline.sh          # Complete LLM + analysis pipeline
├── analyze_only.sh          # Analysis-only pipeline
├── run_llm.py               # LLM code generation
├── run_codeql.py            # CodeQL security analysis
├── config.json              # LLM model and prompt configuration
├── gpu_requirements.txt     # Python dependencies
├── klee_requirements.txt    # System dependencies reference
├── generated_code/          # All generated and processed code
│   ├── generated_code.c    # Raw LLM output
│   ├── clean_code.c        # Cleaned C source
│   ├── clean_code.bc       # LLVM bitcode
│   ├── clean_code.out      # Compiled executable
│   └── Makefile           # Build configuration
├── klee_output/            # KLEE symbolic execution results
│   ├── test*.ktest        # Generated test cases
│   ├── info               # Execution statistics
│   └── *.err              # Error traces (if any)
└── feedback/              # Analysis feedback and reports
```

## ⚙️ Configuration

### LLM Settings (`config.json`)
```json
{
    "MODEL_PATH": "deepseek-ai/deepseek-coder-1.3b-base",
    "max_new_tokens": 512,
    "num_return_sequences": 1,
    "PROMPT": "write a calculator in C"
}
```

Supported models:
- `deepseek-ai/deepseek-coder-1.3b-base` (default)
- `microsoft/DialoGPT-small`
- Any HuggingFace model compatible with transformers

### Environment Paths
All components install to `/scratch/your_username/`:
- **Python Environment**: `/scratch/your_username/klee-venv/`
- **LLVM/Clang**: `/scratch/your_username/llvm-14/`
- **Z3 Solver**: `/scratch/your_username/z3-build/`
- **SQLite**: `/scratch/your_username/sqlite/`
- **KLEE**: `/scratch/your_username/klee/build/bin/klee`
- **CodeQL**: `/scratch/your_username/codeql/`

## 🔍 Understanding the Output

### CodeQL Results
- **Success**: Security vulnerabilities found and reported
- **No issues**: Code passes security analysis
- **Build errors**: C code compilation issues

### KLEE Results
- **Test Cases**: `test*.ktest` files contain concrete input values
- **Coverage**: Paths explored during symbolic execution  
- **Errors**: `*.err` files contain error traces and bug reports
- **Statistics**: Execution time, paths explored, queries generated

### Example KLEE Output
```
✓ KLEE analysis complete: klee_output/
Generated test cases:
  Test files: 3
  Error files: 1

KLEE Statistics:
Elapsed: 00:00:30
KLEE: done: explored paths = 3
KLEE: done: generated tests = 3
```

## 🛠️ Troubleshooting

### Common Issues

1. **Python 3.11 not found**
   ```bash
   # Install Python 3.11 system-wide (ask admin)
   sudo yum install python3.11
   ```

2. **CUDA not available**
   - Check: `nvidia-smi` and `nvcc --version`
   - LLM will fall back to CPU mode (slower)

3. **Build failures during setup**
   - Check internet connection
   - Ensure sufficient disk space (~10GB)
   - Re-run setup script (it's resumable)

4. **KLEE timeout/no results**
   - Code may have infinite loops
   - External function calls (printf, scanf) limit exploration
   - Increase timeout in `analysis.sh`

### Verification Commands

```bash
# Test Python environment
source /scratch/$(whoami)/klee-venv/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Test KLEE
/scratch/$(whoami)/klee/build/bin/klee --version

# Test CodeQL
/scratch/$(whoami)/codeql/codeql version

# Test complete pipeline
./run_pipeline.sh
```

## 📚 Background

This tool combines three powerful techniques:

1. **Large Language Models (LLMs)**: Generate code from natural language descriptions
2. **Static Analysis (CodeQL)**: Find security vulnerabilities without execution
3. **Symbolic Execution (KLEE)**: Explore all possible program paths systematically

The goal is to create a **secure-by-construction** code generation pipeline that:
- Generates functional code with LLMs
- Identifies security issues with static analysis  
- Validates correctness with comprehensive testing

## ✅ Pipeline Status: FULLY FUNCTIONAL

The complete secure code generation pipeline is now **operational** with proper CodeQL security analysis:

### 🎯 What Works
- **✅ LLM Generation**: DeepSeek models generate C code successfully
- **✅ CodeQL Security Analysis**: Full cpp-security-and-quality.qls query suite  
- **✅ KLEE Symbolic Execution**: Comprehensive path exploration and test generation
- **✅ Unified Pipeline**: Single command runs complete workflow
- **✅ Disk Quota Management**: Automatic cache cleanup in /scratch/ space

### 📊 Example Results
Recent pipeline run generated a calculator program and found **17 security findings**:
```
CodeQL Security Analysis Results
================================
Analyzed with: cpp-security-and-quality.qls
Database: /scratch/user/workflow/codeql_db

Findings (17 total):
[NOTE] cpp/missing-check-scanf: Variables read without proper scanf return value checks
```

### 🔄 Complete Workflow Validated
```
DeepSeek LLM → C Calculator Code → CodeQL Analysis → 17 Security Issues Found → KLEE Test Generation
```

All components work together seamlessly to provide actionable security feedback for LLM-generated code.

## Quick Start

1. **Setup Environment**:
   ```bash
   ./prerequisites-setup.sh
   ```

2. **Activate Virtual Environment**:
   ```bash
   source /scratch/{YOUR_USER_ID}/klee-venv/bin/activate
   ```

3. **Configure Prompt** (edit `config.json`):
   ```json
   {
       "PROMPT": "Write a secure C function that safely handles user input"
   }
   ```

4. **Run Complete Pipeline**:
   ```bash
   ./run_pipeline.sh
   ```
   
   Or generate code only:
   ```bash
   python run_llm.py
   ```

For detailed setup instructions, see [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md).

## Key Features

- **Python 3.11 + Virtual Environment**: Clean, isolated environment
- **CUDA Support**: GPU-accelerated PyTorch and llama-cpp-python
- **Modern Libraries**: Updated transformers library (4.57.0+)
- **Automated Setup**: Single script handles all dependencies
