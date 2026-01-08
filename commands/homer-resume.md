# /homer-resume

Resume the HOMER research loop from the last checkpoint.

## Usage

```
/homer-resume [--checkpoint NAME] [--max-iterations N]
```

## Description

Resumes the autonomous research loop from a saved checkpoint. This is useful when:

- The research was manually paused
- An error occurred and was fixed
- You want to continue after reviewing progress
- System was restarted

## Checkpoint System

HOMER automatically creates checkpoints at:

1. **Stage Completion**: When all stories in a stage are done
2. **Quality Gate Pass**: After passing a quality gate
3. **Manual Save**: When explicitly requested
4. **Error Recovery**: Before risky operations

## What's Restored

When resuming, HOMER restores:

- PRD state (completed stories, current progress)
- Database state (all research artifacts)
- Git branch state
- Iteration counter

## Options

- `--checkpoint NAME`: Resume from a specific named checkpoint
- `--max-iterations N`: Maximum iterations before pausing again

## Available Checkpoints

Run `/homer-status` to see available checkpoints:

```
Checkpoints:
  Name                        Created              Stage         Stories
  ────────────────────────────────────────────────────────────────────────
  stage_literature_complete   2024-01-14 10:23     literature    5/32
  stage_hypothesis_complete   2024-01-14 11:15     hypothesis    7/32
  stage_data_complete         2024-01-14 14:32     data          11/32
  manual_checkpoint_1         2024-01-15 09:00     experiment    14/32
  latest                      2024-01-15 14:32     experiment    19/32
```

## Resume Process

```
┌─────────────────────────────────────────────────────────────────┐
│                      RESUME PROCESS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. LOAD CHECKPOINT                                             │
│      ├── Load PRD from checkpoint                                │
│      ├── Verify database consistency                             │
│      └── Verify git state                                        │
│                                                                  │
│   2. VALIDATE STATE                                              │
│      ├── Check all files exist                                   │
│      ├── Verify dependencies                                     │
│      └── Confirm story states                                    │
│                                                                  │
│   3. RESUME LOOP                                                 │
│      └── Continue with next ready story                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Examples

Resume from latest checkpoint:
```
/homer-resume
```

Resume from a specific checkpoint:
```
/homer-resume --checkpoint stage_data_complete
```

Resume with iteration limit:
```
/homer-resume --max-iterations 5
```

## Error Recovery

If resuming after an error:

1. The failed story will be retried
2. Previous successful stories remain completed
3. All artifacts from completed stories are preserved

## See Also

- `/homer-status` - View current status and checkpoints
- `/homer-research` - Start fresh research loop
- `/homer-cancel` - Cancel the current project
