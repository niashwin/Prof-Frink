# /homer-research

Start or resume the autonomous research loop.

## Usage

```
/homer-research [--resume] [--max-iterations N]
```

## Description

This command starts the main HOMER research loop. It autonomously executes user stories from the PRD, invoking appropriate skills and passing through quality gates at each stage.

## The Research Loop

HOMER follows the Ralph pattern for autonomous execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HOMER RESEARCH LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. READ STATE                                                  │
│      ├── Load research_prd.json                                  │
│      ├── Read progress.txt                                       │
│      └── Check database state                                    │
│                                                                  │
│   2. SELECT NEXT STORY                                           │
│      ├── Find ready stories (dependencies met)                   │
│      └── Pick highest priority                                   │
│                                                                  │
│   3. EXECUTE STORY                                               │
│      ├── Invoke required skills                                  │
│      ├── Generate outputs                                        │
│      └── Log actions and learnings                               │
│                                                                  │
│   4. VALIDATE                                                    │
│      ├── Check acceptance criteria                               │
│      └── Run quality gate if stage complete                      │
│                                                                  │
│   5. COMMIT & CHECKPOINT                                         │
│      ├── Git commit progress                                     │
│      ├── Update PRD (mark story passed)                          │
│      └── Create checkpoint in database                           │
│                                                                  │
│   6. REPEAT until all stories complete                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Research Stages

The research proceeds through these stages in order:

1. **Literature** - Search and review relevant papers
2. **Hypothesis** - Refine and generate hypotheses
3. **Data** - Download, explore, and preprocess datasets
4. **Experiment** - Run baseline and proposed experiments
5. **Analysis** - Statistical analysis and significance testing
6. **Visualization** - Generate publication-quality figures
7. **Writing** - Write paper sections
8. **Review** - Self-review and final compilation

## Quality Gates

At each stage transition, a quality gate validates:

| Gate | Key Checks |
|------|------------|
| LiteratureGate | 20+ papers retrieved, 10+ included, multiple sources |
| DataGate | Data downloaded, EDA complete, preprocessing done |
| ExperimentGate | 2+ baselines, proposed method complete, metrics recorded |
| StatisticsGate | Tests performed, effect sizes, confidence intervals |
| WritingGate | All sections present, content complete, figures included |
| FinalGate | All gates passed, reproducibility artifacts present |

## Options

- `--resume`: Resume from last checkpoint (default behavior)
- `--max-iterations N`: Maximum iterations before pausing (default: unlimited)

## Skills Invoked

HOMER automatically invokes skills from claude-scientific-skills:

- `literature-review`, `semantic-scholar`, `openalex`
- `pandas-expert`, `eda`, `scikit-learn`
- `pytorch`, `hyperparameter-tuning`
- `statistical-analysis`, `scipy`
- `matplotlib`, `seaborn`, `publication-figures`
- `academic-writing`, `latex`

## Output

During execution:
- Progress logged to `progress.txt`
- State persisted to `research_state.db`
- Artifacts saved to appropriate directories
- Git commits for each completed story

## Stopping

The loop can be stopped at any time. Use `/homer-resume` to continue from the last checkpoint.

## Example

```
/homer-research --max-iterations 10
```

This runs up to 10 iterations before pausing for review.
