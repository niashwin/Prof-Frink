"""Unit tests for JSON schemas and validators."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from lib.schemas import (
    BaselineRequirements,
    Constraints,
    DatasetConfig,
    PRDMetadata,
    QualityGateConfig,
    QualityGateResult,
    ResearchPRD,
    ResearchTopic,
    UserStory,
    create_empty_prd,
    validate_prd_file,
    validate_topic_json,
)


class TestDatasetConfig:
    """Tests for DatasetConfig model."""

    def test_valid_kaggle_dataset(self):
        """Test valid Kaggle dataset config."""
        config = DatasetConfig(
            source="kaggle",
            identifier="user/dataset-name",
            description="A test dataset"
        )
        assert config.source == "kaggle"
        assert config.identifier == "user/dataset-name"

    def test_valid_huggingface_dataset(self):
        """Test valid HuggingFace dataset config."""
        config = DatasetConfig(
            source="huggingface",
            identifier="org/dataset"
        )
        assert config.source == "huggingface"

    def test_invalid_source(self):
        """Test that invalid source raises error."""
        with pytest.raises(ValidationError):
            DatasetConfig(
                source="invalid_source",
                identifier="test"
            )

    def test_custom_with_url(self):
        """Test custom source with URL."""
        config = DatasetConfig(
            source="custom",
            identifier="my-dataset",
            url="https://example.com/data.csv"
        )
        assert config.url == "https://example.com/data.csv"


class TestConstraints:
    """Tests for Constraints model."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = Constraints()
        assert constraints.max_compute_hours == 24.0
        assert constraints.gpu_required is False
        assert constraints.required_methods == []

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = Constraints(
            max_compute_hours=48.0,
            target_venue="NeurIPS",
            required_methods=["transformer", "attention"],
            gpu_required=True
        )
        assert constraints.max_compute_hours == 48.0
        assert constraints.target_venue == "NeurIPS"
        assert "transformer" in constraints.required_methods

    def test_compute_hours_bounds(self):
        """Test compute hours validation bounds."""
        with pytest.raises(ValidationError):
            Constraints(max_compute_hours=0.0)  # Below minimum

        with pytest.raises(ValidationError):
            Constraints(max_compute_hours=200.0)  # Above maximum


class TestBaselineRequirements:
    """Tests for BaselineRequirements model."""

    def test_default_requirements(self):
        """Test default baseline requirements."""
        req = BaselineRequirements()
        assert req.compare_to_published is True
        assert req.minimum_baselines == 2
        assert req.include_simple_baseline is True

    def test_custom_requirements(self):
        """Test custom baseline requirements."""
        req = BaselineRequirements(
            minimum_baselines=5,
            include_sota=False
        )
        assert req.minimum_baselines == 5
        assert req.include_sota is False

    def test_minimum_baselines_bounds(self):
        """Test minimum baselines validation."""
        with pytest.raises(ValidationError):
            BaselineRequirements(minimum_baselines=0)


class TestResearchTopic:
    """Tests for ResearchTopic model."""

    @pytest.fixture
    def valid_topic_data(self):
        """Return valid topic data."""
        return {
            "title": "Investigating Attention Mechanisms in Language Models",
            "hypothesis": "Multi-head attention with dynamic routing will outperform standard attention mechanisms on long-context tasks by better capturing hierarchical dependencies.",
            "domain": "ML",
            "datasets": [
                {"source": "kaggle", "identifier": "test/dataset"}
            ],
            "research_questions": ["Does dynamic routing improve performance?"]
        }

    def test_valid_topic(self, valid_topic_data):
        """Test creating valid research topic."""
        topic = ResearchTopic(**valid_topic_data)
        assert topic.title == valid_topic_data["title"]
        assert topic.domain == "ML"

    def test_title_too_short(self, valid_topic_data):
        """Test that short title raises error."""
        valid_topic_data["title"] = "Short"
        with pytest.raises(ValidationError):
            ResearchTopic(**valid_topic_data)

    def test_hypothesis_too_short(self, valid_topic_data):
        """Test that short hypothesis raises error."""
        valid_topic_data["hypothesis"] = "Too short."
        with pytest.raises(ValidationError):
            ResearchTopic(**valid_topic_data)

    def test_invalid_domain(self, valid_topic_data):
        """Test that invalid domain raises error."""
        valid_topic_data["domain"] = "INVALID"
        with pytest.raises(ValidationError):
            ResearchTopic(**valid_topic_data)

    def test_empty_datasets(self, valid_topic_data):
        """Test that empty datasets raises error."""
        valid_topic_data["datasets"] = []
        with pytest.raises(ValidationError):
            ResearchTopic(**valid_topic_data)

    def test_research_questions_must_end_with_questionmark(self, valid_topic_data):
        """Test research questions validation."""
        valid_topic_data["research_questions"] = ["This is not a question"]
        with pytest.raises(ValidationError):
            ResearchTopic(**valid_topic_data)

    def test_all_valid_domains(self, valid_topic_data):
        """Test all valid domain values."""
        valid_domains = [
            "ML", "BIOINFORMATICS", "STATISTICS", "CHEMISTRY",
            "PHYSICS", "MEDICINE", "SOCIAL_SCIENCE", "COMPUTER_VISION", "NLP"
        ]
        for domain in valid_domains:
            valid_topic_data["domain"] = domain
            topic = ResearchTopic(**valid_topic_data)
            assert topic.domain == domain


class TestUserStory:
    """Tests for UserStory model."""

    def test_valid_story(self):
        """Test creating valid user story."""
        story = UserStory(
            id="LIT-001",
            title="Search academic databases",
            stage="literature",
            priority=1
        )
        assert story.id == "LIT-001"
        assert story.passes is False

    def test_invalid_id_format(self):
        """Test that invalid ID format raises error."""
        with pytest.raises(ValidationError):
            UserStory(
                id="INVALID-001",
                title="Test",
                stage="literature",
                priority=1
            )

    def test_invalid_stage(self):
        """Test that invalid stage raises error."""
        with pytest.raises(ValidationError):
            UserStory(
                id="LIT-001",
                title="Test",
                stage="invalid_stage",
                priority=1
            )

    def test_priority_bounds(self):
        """Test priority validation."""
        with pytest.raises(ValidationError):
            UserStory(
                id="LIT-001",
                title="Test",
                stage="literature",
                priority=0  # Below minimum
            )

        with pytest.raises(ValidationError):
            UserStory(
                id="LIT-001",
                title="Test",
                stage="literature",
                priority=11  # Above maximum
            )

    def test_story_with_all_fields(self):
        """Test story with all optional fields."""
        story = UserStory(
            id="EXP-001",
            title="Run baseline experiment",
            description="Execute baseline model training",
            stage="experiment",
            skills_required=["scikit-learn", "pytorch"],
            acceptanceCriteria=["Model trained", "Metrics recorded"],
            priority=3,
            dependencies=["DATA-001"],
            outputs=["model.pkl", "results.json"],
            estimated_duration_minutes=60
        )
        assert len(story.skills_required) == 2
        assert len(story.acceptance_criteria) == 2
        assert story.estimated_duration_minutes == 60


class TestPRDMetadata:
    """Tests for PRDMetadata model."""

    def test_default_metadata(self):
        """Test default metadata values."""
        meta = PRDMetadata()
        assert meta.version == "1.0.0"
        assert meta.iteration_count == 0
        assert meta.created_at is not None

    def test_custom_metadata(self):
        """Test custom metadata values."""
        meta = PRDMetadata(
            version="2.0.0",
            iteration_count=5,
            last_checkpoint="checkpoint_001"
        )
        assert meta.version == "2.0.0"
        assert meta.iteration_count == 5


class TestResearchPRD:
    """Tests for ResearchPRD model."""

    @pytest.fixture
    def valid_prd_data(self):
        """Return valid PRD data."""
        return {
            "project": "test-project",
            "branchName": "test-branch",
            "topic": {
                "title": "Test Research on Machine Learning",
                "hypothesis": "This is a test hypothesis that is long enough to pass validation requirements.",
                "domain": "ML",
                "datasets": [{"source": "kaggle", "identifier": "test/data"}]
            },
            "userStories": [
                {
                    "id": "LIT-001",
                    "title": "Search literature",
                    "stage": "literature",
                    "priority": 1
                },
                {
                    "id": "LIT-002",
                    "title": "Review papers",
                    "stage": "literature",
                    "priority": 2,
                    "dependencies": ["LIT-001"]
                }
            ]
        }

    def test_valid_prd(self, valid_prd_data):
        """Test creating valid PRD."""
        prd = ResearchPRD(**valid_prd_data)
        assert prd.project == "test-project"
        assert len(prd.user_stories) == 2

    def test_invalid_branch_name(self, valid_prd_data):
        """Test that invalid branch name raises error."""
        valid_prd_data["branchName"] = "invalid branch name!"
        with pytest.raises(ValidationError):
            ResearchPRD(**valid_prd_data)

    def test_duplicate_story_ids(self, valid_prd_data):
        """Test that duplicate story IDs raise error."""
        valid_prd_data["userStories"].append({
            "id": "LIT-001",  # Duplicate
            "title": "Another story",
            "stage": "literature",
            "priority": 3
        })
        with pytest.raises(ValidationError):
            ResearchPRD(**valid_prd_data)

    def test_unknown_dependency(self, valid_prd_data):
        """Test that unknown dependency raises error."""
        valid_prd_data["userStories"][1]["dependencies"] = ["UNKNOWN-001"]
        with pytest.raises(ValidationError):
            ResearchPRD(**valid_prd_data)

    def test_circular_dependency(self, valid_prd_data):
        """Test that circular dependency raises error."""
        valid_prd_data["userStories"][0]["dependencies"] = ["LIT-002"]
        # LIT-001 depends on LIT-002, LIT-002 depends on LIT-001
        with pytest.raises(ValidationError):
            ResearchPRD(**valid_prd_data)

    def test_get_ready_stories(self, valid_prd_data):
        """Test getting ready stories."""
        prd = ResearchPRD(**valid_prd_data)

        ready = prd.get_ready_stories()
        assert len(ready) == 1
        assert ready[0].id == "LIT-001"

        # Mark LIT-001 as passed
        prd.mark_story_passed("LIT-001")

        ready = prd.get_ready_stories()
        assert len(ready) == 1
        assert ready[0].id == "LIT-002"

    def test_get_next_story(self, valid_prd_data):
        """Test getting next story by priority."""
        prd = ResearchPRD(**valid_prd_data)

        next_story = prd.get_next_story()
        assert next_story.id == "LIT-001"

    def test_get_stories_by_stage(self, valid_prd_data):
        """Test getting stories by stage."""
        prd = ResearchPRD(**valid_prd_data)

        lit_stories = prd.get_stories_by_stage("literature")
        assert len(lit_stories) == 2

        exp_stories = prd.get_stories_by_stage("experiment")
        assert len(exp_stories) == 0

    def test_get_completion_percentage(self, valid_prd_data):
        """Test completion percentage calculation."""
        prd = ResearchPRD(**valid_prd_data)

        assert prd.get_completion_percentage() == 0.0

        prd.mark_story_passed("LIT-001")
        assert prd.get_completion_percentage() == 50.0

        prd.mark_story_passed("LIT-002")
        assert prd.get_completion_percentage() == 100.0

    def test_mark_story_passed(self, valid_prd_data):
        """Test marking story as passed."""
        prd = ResearchPRD(**valid_prd_data)

        prd.mark_story_passed("LIT-001")

        story = next(s for s in prd.user_stories if s.id == "LIT-001")
        assert story.passes is True

    def test_mark_nonexistent_story_raises(self, valid_prd_data):
        """Test marking nonexistent story raises error."""
        prd = ResearchPRD(**valid_prd_data)

        with pytest.raises(ValueError):
            prd.mark_story_passed("NONEXISTENT-001")

    def test_to_json(self, valid_prd_data):
        """Test JSON serialization."""
        prd = ResearchPRD(**valid_prd_data)

        json_str = prd.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["project"] == "test-project"

    def test_from_json(self, valid_prd_data):
        """Test JSON deserialization."""
        prd = ResearchPRD(**valid_prd_data)
        json_str = prd.to_json()

        restored = ResearchPRD.from_json(json_str)

        assert restored.project == prd.project
        assert len(restored.user_stories) == len(prd.user_stories)

    def test_to_file_and_from_file(self, valid_prd_data):
        """Test file I/O."""
        prd = ResearchPRD(**valid_prd_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            prd.to_file(temp_path)
            assert temp_path.exists()

            restored = ResearchPRD.from_file(temp_path)
            assert restored.project == prd.project
        finally:
            temp_path.unlink(missing_ok=True)


class TestQualityGateModels:
    """Tests for quality gate models."""

    def test_quality_gate_result(self):
        """Test QualityGateResult model."""
        result = QualityGateResult(
            gate_name="TestGate",
            passed=True,
            score=0.85,
            threshold=0.7,
            details={"check1": "passed"},
            recommendations=[]
        )
        assert result.passed is True
        assert result.score == 0.85

    def test_quality_gate_result_with_recommendations(self):
        """Test QualityGateResult with recommendations."""
        result = QualityGateResult(
            gate_name="TestGate",
            passed=False,
            score=0.5,
            threshold=0.7,
            recommendations=["Increase sample size", "Add more baselines"]
        )
        assert result.passed is False
        assert len(result.recommendations) == 2

    def test_quality_gate_config(self):
        """Test QualityGateConfig model."""
        config = QualityGateConfig(
            name="ExperimentGate",
            threshold=0.8,
            required=True,
            max_retries=5
        )
        assert config.name == "ExperimentGate"
        assert config.threshold == 0.8


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_prd_file_valid(self):
        """Test validating a valid PRD file."""
        prd_data = {
            "project": "test",
            "branchName": "test-branch",
            "topic": {
                "title": "Test Research Project Title",
                "hypothesis": "This is a sufficiently long hypothesis for validation purposes.",
                "domain": "ML",
                "datasets": [{"source": "kaggle", "identifier": "test/data"}]
            },
            "userStories": [{
                "id": "LIT-001",
                "title": "Test story",
                "stage": "literature",
                "priority": 1
            }]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(prd_data, f)
            temp_path = Path(f.name)

        try:
            is_valid, error = validate_prd_file(temp_path)
            assert is_valid is True
            assert error is None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_prd_file_invalid(self):
        """Test validating an invalid PRD file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "data"}, f)
            temp_path = Path(f.name)

        try:
            is_valid, error = validate_prd_file(temp_path)
            assert is_valid is False
            assert error is not None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_topic_json_valid(self):
        """Test validating valid topic JSON."""
        topic_json = json.dumps({
            "title": "Test Research Project Title",
            "hypothesis": "This is a sufficiently long hypothesis for validation purposes.",
            "domain": "ML",
            "datasets": [{"source": "kaggle", "identifier": "test/data"}]
        })

        is_valid, error = validate_topic_json(topic_json)
        assert is_valid is True
        assert error is None

    def test_validate_topic_json_invalid(self):
        """Test validating invalid topic JSON."""
        topic_json = json.dumps({"title": "Short"})

        is_valid, error = validate_topic_json(topic_json)
        assert is_valid is False
        assert error is not None


class TestCreateEmptyPRD:
    """Tests for create_empty_prd helper."""

    def test_creates_prd_with_minimal_stories(self):
        """Test creating PRD with minimal stories."""
        topic = ResearchTopic(
            title="Test Research Project Title",
            hypothesis="This is a sufficiently long hypothesis for validation purposes.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        prd = create_empty_prd("test-project", topic)

        assert prd.project == "test-project"
        assert prd.branch_name == "test-project"
        assert len(prd.user_stories) >= 1

    def test_branch_name_normalization(self):
        """Test that branch name is normalized."""
        topic = ResearchTopic(
            title="Test Research Project Title",
            hypothesis="This is a sufficiently long hypothesis for validation purposes.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        prd = create_empty_prd("My Test Project", topic)

        assert " " not in prd.branch_name
        assert prd.branch_name == "my-test-project"
