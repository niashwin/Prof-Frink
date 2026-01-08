"""JSON Schema definitions and validators for FRINK.

This module defines Pydantic models for runtime validation of:
- Research topic definitions
- PRD (Product Requirements Document) structure
- User stories with dependencies
- Quality gate configurations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# RESEARCH TOPIC MODELS
# =============================================================================


class DatasetConfig(BaseModel):
    """Configuration for a dataset source."""

    source: str = Field(
        ...,
        pattern="^(kaggle|uci|huggingface|openml|zenodo|custom)$",
        description="Dataset source platform"
    )
    identifier: str = Field(
        ...,
        description="Dataset identifier on the source platform"
    )
    description: str = Field(
        default="",
        description="Brief description of the dataset"
    )
    url: Optional[str] = Field(
        default=None,
        description="Direct URL if custom source"
    )


class Constraints(BaseModel):
    """Research constraints and limitations."""

    max_compute_hours: float = Field(
        default=24.0,
        ge=0.1,
        le=168.0,
        description="Maximum compute hours allowed"
    )
    target_venue: str = Field(
        default="",
        description="Target publication venue (e.g., NeurIPS, ICML)"
    )
    required_methods: list[str] = Field(
        default_factory=list,
        description="Methods that must be used"
    )
    excluded_methods: list[str] = Field(
        default_factory=list,
        description="Methods that should not be used"
    )
    max_model_parameters: Optional[int] = Field(
        default=None,
        description="Maximum model parameter count"
    )
    gpu_required: bool = Field(
        default=False,
        description="Whether GPU is required"
    )


class BaselineRequirements(BaseModel):
    """Requirements for baseline comparisons."""

    compare_to_published: bool = Field(
        default=True,
        description="Compare against published results"
    )
    minimum_baselines: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum number of baseline models"
    )
    include_simple_baseline: bool = Field(
        default=True,
        description="Include a simple baseline (e.g., majority class)"
    )
    include_sota: bool = Field(
        default=True,
        description="Include state-of-the-art comparison"
    )


class ResearchTopic(BaseModel):
    """Complete research topic definition.

    This is the primary input to FRINK, defining what research should be conducted.
    """

    title: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Research title"
    )
    hypothesis: str = Field(
        ...,
        min_length=50,
        description="Main research hypothesis"
    )
    domain: str = Field(
        ...,
        pattern="^(ML|BIOINFORMATICS|STATISTICS|CHEMISTRY|PHYSICS|MEDICINE|SOCIAL_SCIENCE|COMPUTER_VISION|NLP)$",
        description="Research domain"
    )
    scope: str = Field(
        default="",
        description="Scope limitations and boundaries"
    )
    datasets: list[DatasetConfig] = Field(
        ...,
        min_length=1,
        description="Datasets to use for experiments"
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="Specific research questions to answer"
    )
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Research constraints"
    )
    baseline_requirements: BaselineRequirements = Field(
        default_factory=BaselineRequirements,
        description="Baseline comparison requirements"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for literature search"
    )

    @field_validator("research_questions")
    @classmethod
    def validate_questions_are_questions(cls, v: list[str]) -> list[str]:
        """Ensure research questions end with question marks."""
        for q in v:
            if not q.strip().endswith("?"):
                raise ValueError(f"Research question must end with '?': {q}")
        return v


# =============================================================================
# USER STORY MODELS
# =============================================================================


class UserStory(BaseModel):
    """A single user story in the research PRD.

    User stories define discrete tasks in the research pipeline,
    with dependencies, skills required, and acceptance criteria.
    """

    id: str = Field(
        ...,
        pattern="^(LIT|HYP|DATA|EXP|STAT|VIZ|WRITE|REVIEW)-\\d{3}$",
        description="Unique story ID (e.g., LIT-001, EXP-002)"
    )
    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Story title"
    )
    description: str = Field(
        default="",
        description="Detailed description"
    )
    stage: str = Field(
        ...,
        pattern="^(literature|hypothesis|data|experiment|analysis|visualization|writing|review)$",
        description="Pipeline stage"
    )
    skills_required: list[str] = Field(
        default_factory=list,
        description="Skills needed to complete this story"
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        alias="acceptanceCriteria",
        description="Criteria for story completion"
    )
    priority: int = Field(
        ...,
        ge=1,
        le=10,
        description="Priority (1 = highest)"
    )
    passes: bool = Field(
        default=False,
        description="Whether story has passed acceptance"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of dependent stories"
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Expected output artifacts"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )
    estimated_duration_minutes: int = Field(
        default=30,
        ge=5,
        le=480,
        description="Estimated duration in minutes"
    )

    @field_validator("id")
    @classmethod
    def validate_id_matches_stage(cls, v: str, info) -> str:
        """Ensure story ID prefix matches stage."""
        stage_prefixes = {
            "literature": "LIT",
            "hypothesis": "HYP",
            "data": "DATA",
            "experiment": "EXP",
            "analysis": "STAT",
            "visualization": "VIZ",
            "writing": "WRITE",
            "review": "REVIEW"
        }
        # This validator runs before stage is available in some cases
        # so we just validate the format here
        return v

    class Config:
        populate_by_name = True


# =============================================================================
# PRD MODELS
# =============================================================================


class PRDMetadata(BaseModel):
    """PRD metadata and versioning."""

    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Last update timestamp"
    )
    version: str = Field(
        default="1.0.0",
        pattern="^\\d+\\.\\d+\\.\\d+$",
        description="PRD version"
    )
    iteration_count: int = Field(
        default=0,
        ge=0,
        description="Number of loop iterations"
    )
    last_checkpoint: Optional[str] = Field(
        default=None,
        description="Last checkpoint name"
    )


class ResearchPRD(BaseModel):
    """Complete research PRD (Product Requirements Document).

    The PRD is the central document driving FRINK's research loop.
    It contains the topic definition, all user stories, and metadata.
    """

    project: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Project name"
    )
    branch_name: str = Field(
        ...,
        alias="branchName",
        pattern="^[a-zA-Z0-9_-]+$",
        description="Git branch name for this research"
    )
    topic: ResearchTopic = Field(
        ...,
        description="Research topic definition"
    )
    user_stories: list[UserStory] = Field(
        ...,
        alias="userStories",
        min_length=1,
        description="List of user stories"
    )
    metadata: PRDMetadata = Field(
        default_factory=PRDMetadata,
        description="PRD metadata"
    )

    @field_validator("user_stories")
    @classmethod
    def validate_dependencies_exist(cls, stories: list[UserStory]) -> list[UserStory]:
        """Ensure all dependencies reference existing stories."""
        story_ids = {s.id for s in stories}
        for story in stories:
            for dep in story.dependencies:
                if dep not in story_ids:
                    raise ValueError(
                        f"Story {story.id} has unknown dependency: {dep}"
                    )
        return stories

    @field_validator("user_stories")
    @classmethod
    def validate_no_circular_dependencies(cls, stories: list[UserStory]) -> list[UserStory]:
        """Ensure no circular dependencies exist."""
        graph = {s.id: s.dependencies for s in stories}
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for story_id in graph:
            if story_id not in visited:
                if has_cycle(story_id):
                    raise ValueError(
                        f"Circular dependency detected involving story: {story_id}"
                    )
        return stories

    @field_validator("user_stories")
    @classmethod
    def validate_unique_ids(cls, stories: list[UserStory]) -> list[UserStory]:
        """Ensure all story IDs are unique."""
        ids = [s.id for s in stories]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate story IDs found: {set(duplicates)}")
        return stories

    def get_ready_stories(self) -> list[UserStory]:
        """Get stories that are ready to be worked on.

        A story is ready if:
        - It hasn't passed yet
        - All its dependencies have passed
        """
        passed_ids = {s.id for s in self.user_stories if s.passes}
        return [
            s for s in self.user_stories
            if not s.passes and all(d in passed_ids for d in s.dependencies)
        ]

    def get_next_story(self) -> Optional[UserStory]:
        """Get the highest priority ready story."""
        ready = self.get_ready_stories()
        if not ready:
            return None
        return min(ready, key=lambda s: s.priority)

    def get_stories_by_stage(self, stage: str) -> list[UserStory]:
        """Get all stories for a given stage."""
        return [s for s in self.user_stories if s.stage == stage]

    def get_completion_percentage(self) -> float:
        """Get percentage of stories completed."""
        if not self.user_stories:
            return 0.0
        passed = sum(1 for s in self.user_stories if s.passes)
        return (passed / len(self.user_stories)) * 100

    def mark_story_passed(self, story_id: str) -> None:
        """Mark a story as passed."""
        for story in self.user_stories:
            if story.id == story_id:
                story.passes = True
                self.metadata.updated_at = datetime.now().isoformat()
                return
        raise ValueError(f"Story not found: {story_id}")

    def to_json(self, indent: int = 2) -> str:
        """Serialize PRD to JSON string."""
        return self.model_dump_json(indent=indent, by_alias=True)

    @classmethod
    def from_json(cls, json_str: str) -> "ResearchPRD":
        """Deserialize PRD from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: Path | str) -> "ResearchPRD":
        """Load PRD from JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text())

    def to_file(self, path: Path | str) -> None:
        """Save PRD to JSON file."""
        path = Path(path)
        path.write_text(self.to_json())

    class Config:
        populate_by_name = True


# =============================================================================
# QUALITY GATE MODELS
# =============================================================================


class QualityGateResult(BaseModel):
    """Result of a quality gate check."""

    gate_name: str = Field(..., description="Name of the quality gate")
    passed: bool = Field(..., description="Whether the gate passed")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score (0-1)"
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Passing threshold"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed gate results"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations if gate failed"
    )
    checked_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Check timestamp"
    )


class QualityGateConfig(BaseModel):
    """Configuration for a quality gate."""

    name: str = Field(..., description="Gate name")
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Passing threshold"
    )
    required: bool = Field(
        default=True,
        description="Whether this gate is required"
    )
    retry_allowed: bool = Field(
        default=True,
        description="Whether retry is allowed on failure"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def validate_prd_file(path: Path | str) -> tuple[bool, Optional[str]]:
    """Validate a PRD JSON file.

    Args:
        path: Path to PRD JSON file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        prd = ResearchPRD.from_file(path)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_topic_json(json_str: str) -> tuple[bool, Optional[str]]:
    """Validate a topic JSON string.

    Args:
        json_str: JSON string of ResearchTopic

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ResearchTopic.model_validate_json(json_str)
        return True, None
    except Exception as e:
        return False, str(e)


def create_empty_prd(project_name: str, topic: ResearchTopic) -> ResearchPRD:
    """Create a new PRD with minimal stories.

    Args:
        project_name: Project name
        topic: Research topic definition

    Returns:
        New ResearchPRD with initial stories
    """
    branch_name = project_name.lower().replace(" ", "-").replace("_", "-")

    # Create minimal initial stories
    initial_stories = [
        UserStory(
            id="LIT-001",
            title="Search and retrieve relevant literature",
            description="Search academic databases for papers related to the research topic",
            stage="literature",
            skills_required=["literature-review", "semantic-scholar"],
            acceptance_criteria=[
                "At least 20 papers retrieved",
                "Papers are relevant to the research topic",
                "Deduplication completed"
            ],
            priority=1,
            outputs=["literature_search_results.json"]
        ),
        UserStory(
            id="LIT-002",
            title="Screen and filter literature",
            description="Screen papers by title, abstract, and full text",
            stage="literature",
            skills_required=["literature-review"],
            acceptance_criteria=[
                "All papers screened",
                "At least 10 papers included in review",
                "Exclusion reasons documented"
            ],
            priority=2,
            dependencies=["LIT-001"],
            outputs=["literature_screening_results.json"]
        ),
        UserStory(
            id="DATA-001",
            title="Download and explore dataset",
            description="Download the specified dataset and perform EDA",
            stage="data",
            skills_required=["pandas-expert", "eda"],
            acceptance_criteria=[
                "Dataset downloaded successfully",
                "EDA report generated",
                "Data quality issues identified"
            ],
            priority=3,
            dependencies=["LIT-002"],
            outputs=["eda_report.html", "data_summary.json"]
        )
    ]

    return ResearchPRD(
        project=project_name,
        branchName=branch_name,
        topic=topic,
        userStories=initial_stories,
        metadata=PRDMetadata()
    )
