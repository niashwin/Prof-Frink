# /homer-init

Initialize a new HOMER autonomous research project.

## Usage

```
/homer-init <project_name>
```

## Description

This command initializes a new autonomous research project. You will be guided through defining your research topic, including:

- Research title and hypothesis
- Research domain
- Target datasets
- Research constraints
- Baseline requirements

## Process

1. **Project Setup**: Create project directory and initialize database
2. **Topic Definition**: Define the research topic with all required fields
3. **PRD Generation**: Generate the Product Requirements Document with user stories
4. **Validation**: Validate the PRD structure and dependencies
5. **Initial Commit**: Save the initial state

## Arguments

- `project_name`: A unique name for this research project (required)

## Example

```
/homer-init "attention-mechanisms-efficiency"
```

## Research Topic Fields

When defining your research topic, you'll provide:

| Field | Description | Required |
|-------|-------------|----------|
| title | Research title (10-200 chars) | Yes |
| hypothesis | Main research hypothesis (50+ chars) | Yes |
| domain | Research domain (ML, BIOINFORMATICS, etc.) | Yes |
| datasets | List of dataset configurations | Yes |
| research_questions | Specific questions to answer | No |
| constraints | Compute and method constraints | No |
| baseline_requirements | Baseline comparison requirements | No |

## Dataset Configuration

Each dataset requires:
- `source`: kaggle, uci, huggingface, openml, zenodo, or custom
- `identifier`: Dataset identifier on the source platform
- `description`: Brief description

## Output

After initialization:
- `research_prd.json` - Generated PRD with all user stories
- `progress.txt` - Progress tracking file
- `research_state.db` - SQLite database for state persistence
- Project directory structure created

## Next Steps

After initialization, run `/homer-research` to start the autonomous research loop.
