# FRINK API Documentation

This document provides detailed API documentation for FRINK's core modules.

## Table of Contents

- [Research Topic Schema](#research-topic-schema)
- [PRD Generator](#prd-generator)
- [Database Manager](#database-manager)
- [Quality Gates](#quality-gates)
- [Research Loop](#research-loop)

---

## Research Topic Schema

### `ResearchTopic`

The input specification for a research project.

```python
from lib.schemas import ResearchTopic, DatasetConfig, Constraints

topic = ResearchTopic(
    title="Investigating Attention Mechanisms",
    hypothesis="Multi-head attention improves performance on long-context tasks.",
    domain="ML",
    datasets=[
        DatasetConfig(source="kaggle", identifier="squad/question-answering")
    ],
    research_questions=["How does context length affect performance?"],
    constraints=Constraints(max_compute_hours=10.0, gpu_required=True)
)
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | `str` | Yes | Research title (10-200 characters) |
| `hypothesis` | `str` | Yes | Main hypothesis (50+ characters) |
| `domain` | `str` | Yes | Research domain (ML, NLP, MEDICINE, etc.) |
| `datasets` | `list[DatasetConfig]` | Yes | At least one dataset configuration |
| `research_questions` | `list[str]` | No | Optional specific research questions |
| `constraints` | `Constraints` | No | Resource constraints |

**Supported Domains:**
- `ML` - Machine Learning
- `NLP` - Natural Language Processing
- `COMPUTER_VISION` - Computer Vision
- `BIOINFORMATICS` - Bioinformatics
- `STATISTICS` - Statistical Analysis
- `MEDICINE` - Medical/Clinical Research
- `CHEMISTRY` - Computational Chemistry
- `PHYSICS` - Computational Physics
- `SOCIAL_SCIENCE` - Social Science Research

### `DatasetConfig`

Configuration for a dataset source.

```python
DatasetConfig(
    source="kaggle",           # kaggle, uci, huggingface, openml, zenodo, custom
    identifier="user/dataset", # Dataset identifier
    description="Optional description"
)
```

### `Constraints`

Resource constraints for the research project.

```python
Constraints(
    max_compute_hours=10.0,
    gpu_required=True
)
```

---

## PRD Generator

### `generate_prd()`

Generate a PRD from a research topic.

```python
from lib.prd_generator import generate_prd
from lib.schemas import ResearchTopic

topic = ResearchTopic(...)
prd = generate_prd("project-name", topic)

# Access stories
for story in prd.user_stories:
    print(f"{story.id}: {story.title}")

# Get stories by stage
lit_stories = prd.get_stories_by_stage("literature")

# Get next ready story
next_story = prd.get_next_story()

# Mark story complete
prd.mark_story_passed("LIT-001")

# Save to file
prd.to_file("research_prd.json")

# Load from file
prd = ResearchPRD.from_file("research_prd.json")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_name` | `str` | Project identifier |
| `topic` | `ResearchTopic` | Research topic specification |

**Returns:** `ResearchPRD` - A complete PRD with 32 user stories

### `ResearchPRD`

The research PRD containing all user stories.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_stories_by_stage(stage)` | `list[UserStory]` | Get stories for a pipeline stage |
| `get_ready_stories()` | `list[UserStory]` | Get stories with satisfied dependencies |
| `get_next_story()` | `UserStory | None` | Get next story to execute |
| `mark_story_passed(story_id)` | `None` | Mark a story as completed |
| `get_completion_percentage()` | `float` | Get overall completion (0-100) |
| `to_json()` | `str` | Serialize to JSON string |
| `to_file(path)` | `None` | Save to file |
| `from_json(json_str)` | `ResearchPRD` | Load from JSON string |
| `from_file(path)` | `ResearchPRD` | Load from file |

---

## Database Manager

### `DatabaseManager`

Manages SQLite database connections and operations.

```python
from lib.db.manager import DatabaseManager

db = DatabaseManager("research_state.db")
```

### Project Operations

```python
# Create project
project_id = db.create_project("my-research", topic.model_dump_json())

# Get project
project = db.get_project(project_id)
project = db.get_project_by_name("my-research")

# Update project status
db.update_project_status(project_id, "in_progress")

# Update project stage
db.update_project_stage(project_id, "experiment")
```

### Literature Operations

```python
# Insert literature
lit_id = db.insert_literature(
    project_id,
    title="Paper Title",
    source="openalex",
    relevance_score=0.85,
    included_in_review=True
)

# Get literature
lit = db.get_literature(lit_id)
included = db.get_included_literature(project_id)

# Count literature
total = db.count_literature(project_id, included_only=False)
included = db.count_literature(project_id, included_only=True)
```

### Dataset Operations

```python
# Insert dataset
ds_id = db.insert_dataset(
    project_id,
    source="kaggle",
    identifier="user/dataset",
    downloaded=True,
    eda_completed=True,
    preprocessing_completed=True,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)

# Get datasets
datasets = db.get_datasets(project_id)
```

### Experiment Operations

```python
# Insert experiment
exp_id = db.insert_experiment(
    project_id,
    experiment_name="baseline_rf",
    experiment_type="baseline",  # baseline, proposed, ablation
    random_seed=42
)

# Update experiment results
db.update_experiment_results(
    exp_id,
    metrics_json='{"accuracy": 0.85, "f1": 0.82}',
    runtime_seconds=120.5
)

# Get experiments
experiments = db.get_experiments_by_type(project_id, "baseline")
completed = db.get_completed_experiments(project_id)
```

### Statistical Test Operations

```python
# Insert statistical test
test_id = db.insert_statistical_test(
    project_id,
    test_name="paired_t_test",
    effect_size=0.5,
    ci_lower=0.3,
    ci_upper=0.7,
    assumptions_checked=True,
    significant=True
)

# Get statistical tests
tests = db.get_statistical_tests(project_id)
```

### Paper Section Operations

```python
# Insert paper section
section_id = db.insert_paper_section(
    project_id,
    section_name="abstract",
    content="This paper investigates...",
    word_count=250,
    status="draft",
    section_order=1
)

# Get sections
sections = db.get_all_paper_sections(project_id)
abstract = db.get_paper_section(project_id, "abstract")
```

### Figure Operations

```python
# Insert figure
fig_id = db.insert_figure(
    project_id,
    figure_type="results",
    file_path="figures/results.pdf",
    included_in_paper=True
)

# Get figures
figures = db.get_included_figures(project_id)
```

### Checkpoint Operations

```python
# Create checkpoint
cp_id = db.create_checkpoint(
    project_id,
    name="stage_complete_literature",
    prd_json=prd.to_json(),
    progress_txt="Completed literature review...",
    git_hash="abc123"
)

# Get checkpoints
latest = db.get_latest_checkpoint(project_id)
specific = db.get_checkpoint_by_name(project_id, "stage_complete_literature")
all_checkpoints = db.list_checkpoints(project_id)
```

### Agent Logging

```python
# Log action
db.log_agent_action(
    project_id=project_id,
    iteration=1,
    story_id="LIT-001",
    action="Executing literature search",
    action_type="story_start",
    skill_used="literature-search",
    result="success",
    output_summary="Found 50 papers"
)

# Get log
log_entries = db.get_agent_log(project_id, limit=100)
```

---

## Quality Gates

### Gate Types

FRINK includes 6 quality gates:

| Gate | Stage | Key Checks |
|------|-------|------------|
| `LiteratureGate` | literature | 20+ papers, 10+ included, multiple sources |
| `DataGate` | data | Downloaded, EDA complete, splits created |
| `ExperimentGate` | experiment | 2+ baselines, proposed method, seeds set |
| `StatisticsGate` | analysis | Tests performed, effect sizes, CIs |
| `WritingGate` | writing | All sections, content complete, figures |
| `FinalGate` | review | All prior gates, artifacts present |

### Using Quality Gates

```python
from lib.quality_gates import GateManager, LiteratureGate

# Using GateManager
manager = GateManager(db)

# Evaluate a specific stage
result = manager.evaluate_stage(project_id, "literature")
print(f"Passed: {result.passed}, Score: {result.score}")

# Check if can proceed
can_proceed = manager.can_proceed(project_id, "literature", "hypothesis")

# Evaluate all gates
all_results = manager.evaluate_all(project_id)
for stage, result in all_results.items():
    print(f"{stage}: {result.passed}")
```

### Custom Gate Configuration

```python
# Create gate with custom threshold
gate = LiteratureGate(threshold=0.8)  # Default is 0.7

# Evaluate
result = gate.evaluate(project_id, db)

# Check retry capability
if not result.passed and gate.can_retry():
    gate.increment_retry()
    # Retry logic...
```

### Gate Result Structure

```python
result = QualityGateResult(
    gate_name="LiteratureGate",
    passed=True,
    score=0.85,
    threshold=0.7,
    details={
        "checks": [
            {
                "name": "minimum_papers_retrieved",
                "passed": True,
                "score": 1.0,
                "message": "Retrieved 25 papers (minimum: 20)"
            },
            # ... more checks
        ],
        "retry_count": 0
    },
    recommendations=[],  # Empty if passed, suggestions if failed
    checked_at="2024-01-15T10:30:00"
)
```

---

## Research Loop

### `ResearchLoop`

The main orchestrator for autonomous research execution.

```python
from lib.research_loop import ResearchLoop, LoopStatus

# Initialize loop
loop = ResearchLoop(project_id, project_dir, db)

# Register callbacks
loop.on("story_complete", lambda **kw: print(f"Completed: {kw['story_id']}"))
loop.on("stage_complete", lambda **kw: print(f"Stage done: {kw['stage']}"))
loop.on("checkpoint", lambda **kw: print(f"Checkpoint at iteration {kw['iteration']}"))

# Run the loop
status = loop.run(max_iterations=100)

if status == LoopStatus.COMPLETED:
    print("Research completed successfully!")
elif status == LoopStatus.FAILED:
    print("Research failed")
```

### Loop Status

```python
from lib.research_loop import LoopStatus

# Available statuses
LoopStatus.IDLE        # Not started
LoopStatus.RUNNING     # Currently executing
LoopStatus.PAUSED      # Temporarily paused
LoopStatus.COMPLETED   # Successfully finished
LoopStatus.FAILED      # Failed with error
```

### Event Callbacks

Register callbacks for loop events:

```python
# Story completed
loop.on("story_complete", lambda story_id, success, **kw: ...)

# Stage completed (gate passed)
loop.on("stage_complete", lambda stage, **kw: ...)

# Checkpoint created
loop.on("checkpoint", lambda iteration, **kw: ...)

# Error occurred
loop.on("error", lambda error, story_id=None, **kw: ...)
```

### Story Execution

```python
from lib.research_loop import StoryExecutor, StoryResult, SkillInvoker

# Custom skill invoker
class MySkillInvoker(SkillInvoker):
    def invoke(self, skill_name: str, params: dict) -> dict:
        # Custom skill invocation logic
        return {"output": "result"}

# Story executor
executor = StoryExecutor(
    project_dir=Path("./my-project"),
    skill_invoker=MySkillInvoker()
)

# Execute a story
result = executor.execute(story, {"project_dir": "./my-project"})

if result.success:
    print(f"Story {result.story_id} completed")
    print(f"Outputs: {result.outputs}")
else:
    print(f"Failed: {result.error}")
```

### Checkpoint Management

```python
from lib.research_loop import CheckpointManager

cp_manager = CheckpointManager(project_dir, db)

# Create checkpoint
cp_id = cp_manager.create_checkpoint(
    project_id,
    prd,
    checkpoint_type="manual",
    trigger_reason="User requested checkpoint"
)

# Log progress
cp_manager.log_progress("Completed literature review stage")
```

---

## CLI Commands

FRINK provides Claude Code slash commands:

| Command | Description |
|---------|-------------|
| `/frink-init <name>` | Initialize a new research project |
| `/frink-research` | Start the autonomous research loop |
| `/frink-status` | Display current project status |
| `/frink-resume` | Resume from last checkpoint |
| `/frink-cancel` | Cancel the current project |

---

## Error Handling

### Common Exceptions

```python
from pydantic import ValidationError

# Topic validation error
try:
    topic = ResearchTopic(
        title="Too short",  # Will fail validation
        hypothesis="Short",
        domain="ML",
        datasets=[]
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# PRD validation error
try:
    prd = ResearchPRD.from_file("invalid.json")
except (ValidationError, FileNotFoundError) as e:
    print(f"Error loading PRD: {e}")
```

### Database Errors

```python
import sqlite3

try:
    db.create_project("duplicate-name", "{}")
except sqlite3.IntegrityError:
    print("Project name already exists")
```

---

## Best Practices

1. **Always validate input**: Use Pydantic models for validation
2. **Use checkpoints**: Create checkpoints at important milestones
3. **Handle failures gracefully**: Quality gates can retry failed checks
4. **Log progress**: Use the agent log for debugging and auditing
5. **Test with real datasets**: E2E tests use actual Kaggle dataset identifiers
