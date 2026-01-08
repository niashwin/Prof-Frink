# /frink-cancel

Cancel the current FRINK research project.

## Usage

```
/frink-cancel [--force] [--preserve-artifacts]
```

## Description

Cancels the current research project. This command:

1. Stops any running research loop
2. Creates a final checkpoint
3. Updates project status to 'cancelled'
4. Optionally cleans up artifacts

## Confirmation

By default, you'll be asked to confirm cancellation:

```
Are you sure you want to cancel project "attention-mechanisms-efficiency"?

Current Progress:
  Stage: experiment
  Stories Completed: 19/32 (59%)

This will:
  - Stop the research loop
  - Create a final checkpoint
  - Mark project as cancelled

Type 'yes' to confirm:
```

## Options

- `--force`: Skip confirmation prompt
- `--preserve-artifacts`: Keep all generated artifacts (default behavior)

## What Happens

### Before Cancellation
- Final checkpoint created
- All current state saved
- Git commit with cancellation note

### After Cancellation
- Project status set to 'cancelled'
- Research loop stops
- All artifacts preserved (unless explicitly deleted)

## Preserved Artifacts

The following are always preserved:

- `research_prd.json` - Final PRD state
- `research_state.db` - Database with all data
- `progress.txt` - Progress history
- All generated outputs (papers, figures, data)
- Git history

## Resuming After Cancel

A cancelled project can be resumed later:

```
/frink-resume --checkpoint pre_cancel
```

Or start fresh with same topic:
```
/frink-init "project-name-v2"
```

## Example

Cancel with confirmation:
```
/frink-cancel
```

Force cancel without prompt:
```
/frink-cancel --force
```

## See Also

- `/frink-status` - Check current status before cancelling
- `/frink-resume` - Resume a cancelled project
- `/frink-init` - Start a new project
