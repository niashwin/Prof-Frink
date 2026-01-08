"""Core research loop for FRINK.

This module implements the main autonomous research loop following the Ralph pattern:
1. Read state (PRD + progress)
2. Select next story
3. Execute story with skills
4. Validate against acceptance criteria
5. Commit and checkpoint
6. Repeat
"""

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from lib.db.manager import DatabaseManager
from lib.quality_gates import GateManager, QualityGateResult
from lib.schemas import ResearchPRD, UserStory


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class LoopStatus(Enum):
    """Status of the research loop."""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageTransition(Enum):
    """Stage transition events."""
    ENTER = "enter"
    EXIT = "exit"
    GATE_PASS = "gate_pass"
    GATE_FAIL = "gate_fail"


@dataclass
class StoryResult:
    """Result of executing a user story."""
    story_id: str
    success: bool
    outputs: list[str]
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    learnings: Optional[str] = None
    patterns_discovered: Optional[str] = None


@dataclass
class LoopState:
    """Current state of the research loop."""
    prd: ResearchPRD
    iteration: int
    current_story: Optional[UserStory]
    current_stage: str
    status: LoopStatus
    last_gate_result: Optional[QualityGateResult] = None


# =============================================================================
# SKILL INVOKER
# =============================================================================


class SkillInvoker:
    """Invokes skills from claude-scientific-skills."""

    def __init__(self, skills_path: Optional[Path] = None):
        """Initialize skill invoker.

        Args:
            skills_path: Path to claude-scientific-skills directory
        """
        self.skills_path = skills_path or Path("claude-scientific-skills")
        self._skill_cache: dict[str, str] = {}

    def invoke(
        self,
        skill_name: str,
        context: dict[str, Any],
        db: DatabaseManager,
        project_id: int
    ) -> dict[str, Any]:
        """Invoke a skill with context.

        In a real implementation, this would use Claude's tool use
        to invoke the skill. Here we define the interface.

        Args:
            skill_name: Name of skill to invoke
            context: Context data for the skill
            db: Database manager
            project_id: Project ID

        Returns:
            Dict with skill outputs
        """
        # Log skill invocation
        db.log_agent_action(
            project_id=project_id,
            iteration=context.get("iteration", 0),
            story_id=context.get("story_id", ""),
            action=f"Invoking skill: {skill_name}",
            action_type="skill_invoke",
            skill_used=skill_name,
            skill_params_json=json.dumps(context)
        )

        # In actual implementation, this would:
        # 1. Load skill prompt from skills_path
        # 2. Prepare context and invoke Claude
        # 3. Parse and return results

        return {
            "skill": skill_name,
            "status": "invoked",
            "context": context
        }

    def get_skill_prompt(self, skill_name: str) -> Optional[str]:
        """Get the prompt for a skill.

        Args:
            skill_name: Skill name

        Returns:
            Skill prompt content or None
        """
        if skill_name in self._skill_cache:
            return self._skill_cache[skill_name]

        # Try to find skill file
        skill_file = self.skills_path / f"{skill_name}.md"
        if skill_file.exists():
            content = skill_file.read_text()
            self._skill_cache[skill_name] = content
            return content

        # Try nested directory structure
        for subdir in self.skills_path.iterdir():
            if subdir.is_dir():
                nested_file = subdir / f"{skill_name}.md"
                if nested_file.exists():
                    content = nested_file.read_text()
                    self._skill_cache[skill_name] = content
                    return content

        return None


# =============================================================================
# STORY EXECUTOR
# =============================================================================


class StoryExecutor:
    """Executes individual user stories."""

    def __init__(
        self,
        db: DatabaseManager,
        skill_invoker: SkillInvoker,
        project_path: Path
    ):
        """Initialize story executor.

        Args:
            db: Database manager
            skill_invoker: Skill invoker
            project_path: Project directory path
        """
        self.db = db
        self.skill_invoker = skill_invoker
        self.project_path = project_path

    def execute(
        self,
        story: UserStory,
        project_id: int,
        iteration: int,
        prd: ResearchPRD
    ) -> StoryResult:
        """Execute a user story.

        Args:
            story: Story to execute
            project_id: Project ID
            iteration: Current iteration
            prd: Current PRD state

        Returns:
            StoryResult with execution outcome
        """
        start_time = time.time()
        outputs = []
        error_message = None
        learnings = None

        try:
            # Log story start
            self.db.log_agent_action(
                project_id=project_id,
                iteration=iteration,
                story_id=story.id,
                action=f"Starting story: {story.title}",
                action_type="story_start"
            )

            # Execute each required skill
            for skill_name in story.skills_required:
                context = self._build_skill_context(story, prd, iteration)
                result = self.skill_invoker.invoke(
                    skill_name,
                    context,
                    self.db,
                    project_id
                )

                # Collect outputs
                if result.get("outputs"):
                    outputs.extend(result["outputs"])

                # Collect learnings
                if result.get("learnings"):
                    learnings = result.get("learnings")

            # Verify acceptance criteria
            criteria_met = self._verify_acceptance_criteria(story, outputs)

            if not criteria_met:
                error_message = "Not all acceptance criteria met"

            duration = time.time() - start_time

            return StoryResult(
                story_id=story.id,
                success=criteria_met,
                outputs=outputs,
                error_message=error_message,
                duration_seconds=duration,
                learnings=learnings
            )

        except Exception as e:
            duration = time.time() - start_time
            return StoryResult(
                story_id=story.id,
                success=False,
                outputs=outputs,
                error_message=str(e),
                duration_seconds=duration
            )

    def _build_skill_context(
        self,
        story: UserStory,
        prd: ResearchPRD,
        iteration: int
    ) -> dict[str, Any]:
        """Build context for skill invocation."""
        return {
            "story_id": story.id,
            "story_title": story.title,
            "story_description": story.description,
            "stage": story.stage,
            "iteration": iteration,
            "topic": prd.topic.model_dump(),
            "acceptance_criteria": story.acceptance_criteria,
            "expected_outputs": story.outputs,
            "project_path": str(self.project_path)
        }

    def _verify_acceptance_criteria(
        self,
        story: UserStory,
        outputs: list[str]
    ) -> bool:
        """Verify that acceptance criteria are met.

        In a real implementation, this would use Claude to verify
        each criterion. Here we provide a basic implementation.

        Args:
            story: Story with acceptance criteria
            outputs: Generated outputs

        Returns:
            True if criteria are met
        """
        # Basic check: all expected outputs generated
        expected = set(story.outputs)
        generated = set(outputs)

        if not expected.issubset(generated):
            return False

        # Would add more sophisticated verification here
        return True


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """Manages checkpoints and state persistence."""

    def __init__(self, db: DatabaseManager, project_path: Path):
        """Initialize checkpoint manager.

        Args:
            db: Database manager
            project_path: Project directory
        """
        self.db = db
        self.project_path = project_path
        self.prd_path = project_path / "research_prd.json"
        self.progress_path = project_path / "progress.txt"

    def save_prd(self, prd: ResearchPRD) -> None:
        """Save PRD to file."""
        prd.to_file(self.prd_path)

    def load_prd(self) -> ResearchPRD:
        """Load PRD from file."""
        return ResearchPRD.from_file(self.prd_path)

    def update_progress(self, message: str) -> None:
        """Append to progress file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def create_checkpoint(
        self,
        project_id: int,
        prd: ResearchPRD,
        checkpoint_type: str = "automatic",
        trigger_reason: Optional[str] = None
    ) -> int:
        """Create a checkpoint.

        Args:
            project_id: Project ID
            prd: Current PRD state
            checkpoint_type: Type of checkpoint
            trigger_reason: Why checkpoint was created

        Returns:
            Checkpoint ID
        """
        # Read progress file
        progress_txt = ""
        if self.progress_path.exists():
            progress_txt = self.progress_path.read_text()

        # Get git hash if available
        git_hash = self._get_git_hash()

        # Calculate progress
        completed = sum(1 for s in prd.user_stories if s.passes)
        total = len(prd.user_stories)

        # Generate checkpoint name with microseconds for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{checkpoint_type}_{timestamp}"

        return self.db.create_checkpoint(
            project_id=project_id,
            name=name,
            prd_json=prd.to_json(),
            progress_txt=progress_txt,
            git_hash=git_hash,
            checkpoint_type=checkpoint_type,
            stories_completed=completed,
            stories_total=total,
            trigger_reason=trigger_reason
        )

    def restore_checkpoint(
        self,
        project_id: int,
        checkpoint_name: Optional[str] = None
    ) -> Optional[ResearchPRD]:
        """Restore from a checkpoint.

        Args:
            project_id: Project ID
            checkpoint_name: Specific checkpoint or latest

        Returns:
            Restored PRD or None
        """
        if checkpoint_name:
            checkpoint = self.db.get_checkpoint_by_name(project_id, checkpoint_name)
        else:
            checkpoint = self.db.get_latest_checkpoint(project_id)

        if not checkpoint:
            return None

        # Restore PRD
        prd = ResearchPRD.from_json(checkpoint["prd_json"])

        # Restore progress file
        if checkpoint.get("progress_txt"):
            with open(self.progress_path, "w") as f:
                f.write(checkpoint["progress_txt"])

        return prd

    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def git_commit(self, message: str) -> bool:
        """Commit current changes.

        Args:
            message: Commit message

        Returns:
            True if successful
        """
        try:
            # Add all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_path,
                check=True
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_path,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


# =============================================================================
# RESEARCH LOOP
# =============================================================================


class ResearchLoop:
    """Main autonomous research loop."""

    def __init__(
        self,
        project_id: int,
        project_path: Path,
        db: DatabaseManager,
        skills_path: Optional[Path] = None
    ):
        """Initialize research loop.

        Args:
            project_id: Project ID in database
            project_path: Project directory path
            db: Database manager
            skills_path: Path to skills directory
        """
        self.project_id = project_id
        self.project_path = project_path
        self.db = db

        self.skill_invoker = SkillInvoker(skills_path)
        self.story_executor = StoryExecutor(db, self.skill_invoker, project_path)
        self.checkpoint_manager = CheckpointManager(db, project_path)
        self.gate_manager = GateManager(db)

        self._status = LoopStatus.PAUSED
        self._iteration = 0
        self._callbacks: dict[str, list[Callable]] = {
            "story_start": [],
            "story_complete": [],
            "stage_transition": [],
            "gate_check": [],
            "checkpoint": [],
            "error": []
        }

    @property
    def status(self) -> LoopStatus:
        """Get current loop status."""
        return self._status

    def register_callback(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """Register a callback for an event.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, **kwargs) -> None:
        """Emit an event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                pass  # Don't let callbacks break the loop

    def run(self, max_iterations: Optional[int] = None) -> LoopStatus:
        """Run the research loop.

        Args:
            max_iterations: Maximum iterations (None for unlimited)

        Returns:
            Final loop status
        """
        self._status = LoopStatus.RUNNING

        # Load current PRD
        prd = self.checkpoint_manager.load_prd()

        # Get starting iteration from project
        project = self.db.get_project(self.project_id)
        self._iteration = project.get("iteration_count", 0)

        iterations_run = 0

        while self._status == LoopStatus.RUNNING:
            # Check iteration limit
            if max_iterations and iterations_run >= max_iterations:
                self._status = LoopStatus.PAUSED
                self.checkpoint_manager.update_progress(
                    f"Paused after {iterations_run} iterations (limit reached)"
                )
                break

            # Get next story
            next_story = prd.get_next_story()

            if not next_story:
                # All stories complete
                self._status = LoopStatus.COMPLETED
                self.checkpoint_manager.update_progress("All stories completed!")
                break

            # Check if stage changed
            current_stage = next_story.stage
            prev_stage = self._get_previous_stage(prd, next_story)

            if prev_stage and prev_stage != current_stage:
                # Stage transition - check quality gate
                gate_result = self._check_quality_gate(prev_stage)
                self._emit("gate_check", result=gate_result, stage=prev_stage)

                if not gate_result.passed:
                    # Gate failed - handle retry or fail
                    gate = self.gate_manager.get_gate(prev_stage)
                    if gate and gate.can_retry():
                        gate.increment_retry()
                        self.checkpoint_manager.update_progress(
                            f"Quality gate {prev_stage} failed, retrying..."
                        )
                        continue
                    else:
                        self._status = LoopStatus.FAILED
                        self.checkpoint_manager.update_progress(
                            f"Quality gate {prev_stage} failed permanently"
                        )
                        break

                self._emit(
                    "stage_transition",
                    from_stage=prev_stage,
                    to_stage=current_stage,
                    transition=StageTransition.GATE_PASS
                )

            # Execute story
            self._emit("story_start", story=next_story)
            self._iteration += 1
            self.db.increment_iteration(self.project_id)

            result = self.story_executor.execute(
                next_story,
                self.project_id,
                self._iteration,
                prd
            )

            if result.success:
                # Mark story as passed
                prd.mark_story_passed(next_story.id)

                # Log completion
                self.db.log_agent_action(
                    project_id=self.project_id,
                    iteration=self._iteration,
                    story_id=next_story.id,
                    action=f"Story completed: {next_story.title}",
                    action_type="complete",
                    result="success",
                    duration_seconds=result.duration_seconds,
                    learnings=result.learnings,
                    patterns_discovered=result.patterns_discovered
                )

                self._emit("story_complete", story=next_story, result=result)

                # Save progress
                self.checkpoint_manager.save_prd(prd)
                self.checkpoint_manager.update_progress(
                    f"Completed: {next_story.id} - {next_story.title}"
                )

                # Git commit
                self.checkpoint_manager.git_commit(
                    f"Complete {next_story.id}: {next_story.title}"
                )

                # Create checkpoint periodically
                if self._iteration % 5 == 0:
                    self.checkpoint_manager.create_checkpoint(
                        self.project_id,
                        prd,
                        checkpoint_type="automatic",
                        trigger_reason=f"Periodic checkpoint at iteration {self._iteration}"
                    )
                    self._emit("checkpoint", iteration=self._iteration)

            else:
                # Story failed
                self.db.log_agent_action(
                    project_id=self.project_id,
                    iteration=self._iteration,
                    story_id=next_story.id,
                    action=f"Story failed: {next_story.title}",
                    action_type="error",
                    result="failed",
                    error_message=result.error_message
                )

                self._emit("error", story=next_story, error=result.error_message)

                # Could implement retry logic here
                self._status = LoopStatus.FAILED
                self.checkpoint_manager.update_progress(
                    f"Failed: {next_story.id} - {result.error_message}"
                )
                break

            iterations_run += 1

        # Final checkpoint
        self.checkpoint_manager.create_checkpoint(
            self.project_id,
            prd,
            checkpoint_type="stage_complete" if self._status == LoopStatus.COMPLETED else "manual",
            trigger_reason=f"Loop ended with status: {self._status.value}"
        )

        # Update project status
        status_map = {
            LoopStatus.COMPLETED: "completed",
            LoopStatus.FAILED: "failed",
            LoopStatus.PAUSED: "paused",
            LoopStatus.CANCELLED: "failed"
        }
        self.db.update_project_status(
            self.project_id,
            status_map.get(self._status, "paused")
        )

        return self._status

    def pause(self) -> None:
        """Pause the research loop."""
        self._status = LoopStatus.PAUSED

    def cancel(self) -> None:
        """Cancel the research loop."""
        self._status = LoopStatus.CANCELLED

    def _get_previous_stage(
        self,
        prd: ResearchPRD,
        current_story: UserStory
    ) -> Optional[str]:
        """Get the stage of the most recently completed story."""
        completed_stories = [s for s in prd.user_stories if s.passes]
        if not completed_stories:
            return None

        # Get most recent (by priority/order)
        latest = max(completed_stories, key=lambda s: s.priority)
        return latest.stage

    def _check_quality_gate(self, stage: str) -> QualityGateResult:
        """Check quality gate for a stage."""
        return self.gate_manager.evaluate_stage(self.project_id, stage)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def start_research(
    project_id: int,
    project_path: Path,
    db: DatabaseManager,
    max_iterations: Optional[int] = None
) -> LoopStatus:
    """Start the research loop.

    Args:
        project_id: Project ID
        project_path: Project directory
        db: Database manager
        max_iterations: Optional iteration limit

    Returns:
        Final loop status
    """
    loop = ResearchLoop(project_id, project_path, db)
    return loop.run(max_iterations)


def resume_research(
    project_id: int,
    project_path: Path,
    db: DatabaseManager,
    checkpoint_name: Optional[str] = None,
    max_iterations: Optional[int] = None
) -> LoopStatus:
    """Resume research from checkpoint.

    Args:
        project_id: Project ID
        project_path: Project directory
        db: Database manager
        checkpoint_name: Specific checkpoint to resume from
        max_iterations: Optional iteration limit

    Returns:
        Final loop status
    """
    loop = ResearchLoop(project_id, project_path, db)

    # Restore from checkpoint
    checkpoint_manager = CheckpointManager(db, project_path)
    prd = checkpoint_manager.restore_checkpoint(project_id, checkpoint_name)

    if prd:
        checkpoint_manager.save_prd(prd)

    return loop.run(max_iterations)
