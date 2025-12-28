# Local AI Assistant

An offline-first, privacy-focused AI assistant that runs entirely on your local machine.

## Core Principles

- 🚫 **Offline-first**: No cloud dependencies, no external APIs
- 🔒 **Local Trust**: Your data stays on your device
- 🧠 **Honest Intelligence**: Transparent about knowledge boundaries
- 🧩 **Modular Design**: Swap components easily
- 👥 **Human-aligned UX**: Interruptible, explainable, and user-controlled

## Architecture

```
Microphone → STT (Whisper.cpp) → Intent/Policy → Conversation Manager → Local LLM (llama.cpp) → Tools/Memory → TTS → Speaker
```

## Getting Started

### Prerequisites

- Python 3.9+
- C++17 compiler
- CMake 3.15+
- Git LFS (for model files)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gpt-k

# Install Python dependencies
pip install -r requirements.txt

# Build native components
./scripts/build.sh
```

## Project Structure

- `src/` - Main source code
  - `audio/` - Audio I/O and processing
  - `llm/` - Language model interface
  - `memory/` - Memory system
  - `tools/` - Built-in tools
  - `ui/` - User interface
  - `utils/` - Utility functions
- `models/` - Local model files
- `config/` - Configuration files
- `scripts/` - Build and utility scripts
- `tests/` - Test suite

## License

MIT License - See [LICENSE](LICENSE) for details.
