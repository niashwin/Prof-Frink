<p align="center">
  <img src="assets/frink-professor.png" alt="FRINK - Autonomous Research Agent" width="300"/>
  <br><br>
  <em>"Glavin! The research, it does itself!"</em>
</p>

<h1 align="center">FRINK</h1>
<h3 align="center">Fully Autonomous Research Intelligence & Knowledge</h3>
<h4 align="center"><em>Named after Professor Frink, Springfield's resident scientist</em></h4>

<p align="center">
  <strong>An autonomous AI agent for end-to-end scientific research</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Alpha Status"/>
</p>

---

## Overview

**FRINK** is an autonomous research agent that takes a scientific hypothesis and dataset specification as input, then autonomously conducts the entire research pipeline—from literature review to publication-ready paper generation.

Built on the [Ralph](https://github.com/snarktank/ralph) autonomous loop pattern and leveraging [Claude Scientific Skills](https://github.com/K-Dense-AI/claude-scientific-skills), FRINK orchestrates complex multi-stage research workflows without human intervention.

## Features

- **Fully Autonomous Pipeline**: From topic definition to paper generation
- **8-Stage Research Workflow**: Literature → Hypothesis → Data → Experiment → Analysis → Visualization → Writing → Review
- **Quality Gates**: Automated validation at each stage transition
- **PRD-Driven Execution**: 32 user stories with dependency management
- **State Persistence**: SQLite database with checkpoint/resume capability
- **Skill Integration**: 139+ scientific skills for specialized tasks
- **Claude Code Plugin**: Native integration with Claude Code CLI

## Quick Start

### 1. Define Your Research Topic

Create a `topic.json` file:

```json
{
  "title": "Investigating Attention Mechanisms in Transformer Models",
  "hypothesis": "Multi-head attention with dynamic routing will outperform standard attention on long-context tasks.",
  "domain": "ML",
  "datasets": [
    {
      "source": "kaggle",
      "identifier": "squad/question-answering"
    }
  ],
  "research_questions": [
    "Does dynamic routing improve attention efficiency?",
    "How does context length affect performance gains?"
  ]
}
```

### 2. Initialize the Project

```bash
/frink-init "attention-mechanisms"
```

### 3. Start Autonomous Research

```bash
/frink-research
```

### 4. Monitor Progress

```bash
/frink-status --detailed
```

## Installation

### Prerequisites

- Python 3.12+
- Claude Code CLI
- Git

### Install FRINK

```bash
# Clone the repository
git clone https://github.com/niashwin/homer.git
cd homer

# Install dependencies
pip install -e .

# Install as Claude Code plugin
claude plugins install .
```

### Install Scientific Skills

```bash
# Clone the skills repository
git clone https://github.com/K-Dense-AI/claude-scientific-skills.git

# Link to FRINK
export FRINK_SKILLS_PATH="./claude-scientific-skills"
```

## Usage

### Commands

| Command | Description |
|---------|-------------|
| `/frink-init <name>` | Initialize a new research project |
| `/frink-research` | Start the autonomous research loop |
| `/frink-status` | Display current project status |
| `/frink-resume` | Resume from last checkpoint |
| `/frink-cancel` | Cancel the current project |

### Research Topic Schema

```python
{
  "title": str,           # Research title (10-200 chars)
  "hypothesis": str,      # Main hypothesis (50+ chars)
  "domain": str,          # ML, NLP, BIOINFORMATICS, etc.
  "datasets": [           # At least one dataset
    {
      "source": str,      # kaggle, huggingface, uci, etc.
      "identifier": str   # Dataset identifier
    }
  ],
  "research_questions": [str],  # Optional specific questions
  "constraints": {
    "max_compute_hours": float,
    "gpu_required": bool
  }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRINK SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  topic.json  │───▶│ PRD Generator │───▶│ research_prd │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RESEARCH LOOP                          │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │   │
│  │  │  Read  │─▶│ Select │─▶│Execute │─▶│Validate│─┐        │   │
│  │  │ State  │  │ Story  │  │ Skills │  │  Gate  │ │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘ │        │   │
│  │       ▲                                         │        │   │
│  │       └─────────────────────────────────────────┘        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Database   │    │  Checkpoints │    │  Git Commits │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Research Stages

| Stage | Gate | Key Tasks |
|-------|------|-----------|
| Literature | LiteratureGate | Search, screen, synthesize papers |
| Hypothesis | - | Refine and formalize hypotheses |
| Data | DataGate | Download, EDA, preprocessing |
| Experiment | ExperimentGate | Baselines, proposed method, ablations |
| Analysis | StatisticsGate | Significance tests, effect sizes |
| Visualization | - | Figures, tables, diagrams |
| Writing | WritingGate | All paper sections |
| Review | FinalGate | Technical review, compilation |

### Quality Gates

Each gate validates stage completion before proceeding:

- **LiteratureGate**: ≥20 papers retrieved, ≥10 included, multiple sources
- **DataGate**: Downloaded, EDA complete, train/val/test splits
- **ExperimentGate**: ≥2 baselines, proposed method, reproducibility seeds
- **StatisticsGate**: Significance tests, effect sizes, confidence intervals
- **WritingGate**: All sections present, content complete, figures included
- **FinalGate**: All prior gates passed, reproducibility verified

## Project Structure

```
frink/
├── .claude-plugin/
│   └── plugin.json          # Claude Code plugin manifest
├── commands/                 # CLI command definitions
│   ├── frink-init.md
│   ├── frink-research.md
│   ├── frink-status.md
│   ├── frink-resume.md
│   └── frink-cancel.md
├── lib/                      # Core library
│   ├── db/
│   │   ├── schema.sql       # SQLite schema
│   │   └── manager.py       # Database manager
│   ├── schemas.py           # Pydantic models
│   ├── prd_generator.py     # PRD generation
│   ├── quality_gates.py     # Quality validation
│   └── research_loop.py     # Main loop orchestrator
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── hooks/
│   └── hooks.json           # Pre/post tool hooks
├── pyproject.toml           # Project configuration
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Supported Domains

- **ML** - Machine Learning
- **NLP** - Natural Language Processing
- **COMPUTER_VISION** - Computer Vision
- **BIOINFORMATICS** - Bioinformatics
- **STATISTICS** - Statistical Analysis
- **MEDICINE** - Medical/Clinical Research
- **CHEMISTRY** - Computational Chemistry
- **PHYSICS** - Computational Physics
- **SOCIAL_SCIENCE** - Social Science Research

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/niashwin/homer.git
cd homer
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=frink --cov-report=html

# Type checking
mypy frink/

# Linting
ruff check frink/
```

## Acknowledgments

FRINK builds upon the excellent work of:

- **[Ralph](https://github.com/snarktank/ralph)** - Autonomous AI agent loop pattern
- **[Claude Scientific Skills](https://github.com/K-Dense-AI/claude-scientific-skills)** - Scientific skill library

## Roadmap

- [ ] Multi-agent collaboration
- [ ] Real-time experiment monitoring
- [ ] ArXiv submission automation
- [ ] Code artifact generation
- [ ] Interactive hypothesis refinement

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with science and determination... and the occasional "Glavin!"</sub>
  <br>
  <sub>Part of the Simpsons-verse autonomous agent family, alongside <a href="https://github.com/snarktank/ralph">Ralph</a></sub>
</p>
