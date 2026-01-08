# /frink-status

Display the current status of the FRINK research project.

## Usage

```
/frink-status [--detailed] [--stage STAGE]
```

## Description

Shows comprehensive status information about the current research project, including:

- Overall progress percentage
- Current stage and story
- Completed and pending stories
- Quality gate results
- Recent activity

## Output Sections

### Project Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                    FRINK PROJECT STATUS                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Project: attention-mechanisms-efficiency                         ║
║  Status:  IN_PROGRESS                                            ║
║  Stage:   experiment                                             ║
║  Story:   EXP-003 (Implement proposed method)                    ║
║                                                                   ║
║  Progress: ████████████░░░░░░░░ 58% (19/32 stories)              ║
║  Iteration: 47                                                    ║
║  Last Activity: 2024-01-15 14:32:00                              ║
╚══════════════════════════════════════════════════════════════════╝
```

### Stage Progress

```
Stages:
  ✓ literature    [5/5 stories] - LiteratureGate PASSED (0.85)
  ✓ hypothesis    [2/2 stories] - HypothesisGate PASSED (0.90)
  ✓ data          [4/4 stories] - DataGate PASSED (0.88)
  → experiment    [3/6 stories] - In Progress
  ○ analysis      [0/3 stories] - Pending
  ○ visualization [0/4 stories] - Pending
  ○ writing       [0/6 stories] - Pending
  ○ review        [0/3 stories] - Pending
```

### Quality Gate Summary

```
Quality Gates:
  Gate              Score    Threshold    Status
  ─────────────────────────────────────────────────
  LiteratureGate    0.85     0.70         ✓ PASSED
  DataGate          0.88     0.80         ✓ PASSED
  ExperimentGate    0.45     0.75         ○ PENDING
  StatisticsGate    0.00     0.70         ○ PENDING
  WritingGate       0.00     0.80         ○ PENDING
  FinalGate         0.00     0.85         ○ PENDING
```

## Options

- `--detailed`: Show detailed information including all stories and their status
- `--stage STAGE`: Show detailed status for a specific stage only

## Detailed View

With `--detailed`, shows each story:

```
Stage: experiment
  ✓ EXP-001  Implement simple baseline           [completed]
  ✓ EXP-002  Implement standard ML baselines     [completed]
  → EXP-003  Implement proposed method           [in_progress]
  ○ EXP-004  Hyperparameter tuning               [pending]
  ○ EXP-005  Run final evaluation on test set    [pending]
  ○ EXP-006  Perform ablation studies            [pending]
```

## Database Statistics

```
Database Stats:
  Papers Retrieved:    127
  Papers Included:     23
  Datasets:           1
  Experiments Run:    5
  Statistical Tests:  0
  Paper Sections:     0
  Figures Generated:  0
```

## Recent Activity

```
Recent Activity (last 10):
  [14:32:00] EXP-003 started - Implementing proposed method
  [14:28:15] EXP-002 completed - Standard baselines done
  [14:15:42] ExperimentGate skipped - Not at stage end
  [14:12:33] EXP-001 completed - Simple baseline done
  ...
```

## Example

```
/frink-status --detailed --stage experiment
```

Shows detailed experiment stage status with all stories and their acceptance criteria.
