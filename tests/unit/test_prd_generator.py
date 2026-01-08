"""Unit tests for PRD generator."""

import pytest

from homer.lib.prd_generator import (
    ANALYSIS_STORIES,
    DATA_STORIES,
    EXPERIMENT_STORIES,
    HYPOTHESIS_STORIES,
    LITERATURE_STORIES,
    REVIEW_STORIES,
    VISUALIZATION_STORIES,
    WRITING_STORIES,
    PRDGenerator,
    generate_prd,
)
from homer.lib.schemas import (
    BaselineRequirements,
    Constraints,
    DatasetConfig,
    ResearchTopic,
)


@pytest.fixture
def sample_topic():
    """Create a sample research topic for testing."""
    return ResearchTopic(
        title="Investigating Neural Attention Mechanisms in NLP",
        hypothesis="Multi-head attention with adaptive routing will improve performance on long-context language tasks by dynamically allocating attention based on input complexity.",
        domain="NLP",
        datasets=[
            DatasetConfig(
                source="kaggle",
                identifier="squad/question-answering",
                description="Question answering dataset"
            ),
            DatasetConfig(
                source="huggingface",
                identifier="wikitext",
                description="Language modeling dataset"
            )
        ],
        research_questions=[
            "Does adaptive routing improve attention efficiency?",
            "How does context length affect performance gains?"
        ],
        constraints=Constraints(
            max_compute_hours=48.0,
            target_venue="ACL",
            gpu_required=True
        ),
        baseline_requirements=BaselineRequirements(
            minimum_baselines=3,
            include_sota=True
        ),
        keywords=["attention", "transformer", "NLP", "language model"]
    )


class TestStoryTemplates:
    """Tests for story template definitions."""

    def test_literature_stories_have_required_fields(self):
        """Test all literature stories have required fields."""
        for story in LITERATURE_STORIES:
            assert "id" in story
            assert story["id"].startswith("LIT-")
            assert "title" in story
            assert "priority" in story
            assert "skills_required" in story

    def test_hypothesis_stories_have_required_fields(self):
        """Test all hypothesis stories have required fields."""
        for story in HYPOTHESIS_STORIES:
            assert "id" in story
            assert story["id"].startswith("HYP-")
            assert "title" in story

    def test_data_stories_have_required_fields(self):
        """Test all data stories have required fields."""
        for story in DATA_STORIES:
            assert "id" in story
            assert story["id"].startswith("DATA-")
            assert "title" in story

    def test_experiment_stories_have_required_fields(self):
        """Test all experiment stories have required fields."""
        for story in EXPERIMENT_STORIES:
            assert "id" in story
            assert story["id"].startswith("EXP-")
            assert "title" in story

    def test_analysis_stories_have_required_fields(self):
        """Test all analysis stories have required fields."""
        for story in ANALYSIS_STORIES:
            assert "id" in story
            assert story["id"].startswith("STAT-")
            assert "title" in story

    def test_visualization_stories_have_required_fields(self):
        """Test all visualization stories have required fields."""
        for story in VISUALIZATION_STORIES:
            assert "id" in story
            assert story["id"].startswith("VIZ-")
            assert "title" in story

    def test_writing_stories_have_required_fields(self):
        """Test all writing stories have required fields."""
        for story in WRITING_STORIES:
            assert "id" in story
            assert story["id"].startswith("WRITE-")
            assert "title" in story

    def test_review_stories_have_required_fields(self):
        """Test all review stories have required fields."""
        for story in REVIEW_STORIES:
            assert "id" in story
            assert story["id"].startswith("REVIEW-")
            assert "title" in story

    def test_story_dependencies_are_valid(self):
        """Test that all story dependencies reference valid stories."""
        all_stories = (
            LITERATURE_STORIES +
            HYPOTHESIS_STORIES +
            DATA_STORIES +
            EXPERIMENT_STORIES +
            ANALYSIS_STORIES +
            VISUALIZATION_STORIES +
            WRITING_STORIES +
            REVIEW_STORIES
        )
        all_ids = {story["id"] for story in all_stories}

        for story in all_stories:
            for dep in story.get("dependencies", []):
                assert dep in all_ids, f"Story {story['id']} has unknown dependency: {dep}"


class TestPRDGenerator:
    """Tests for PRDGenerator class."""

    def test_generate_creates_valid_prd(self, sample_topic):
        """Test that generator creates a valid PRD."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        assert prd.project == "test-project"
        assert prd.topic == sample_topic
        assert len(prd.user_stories) > 0

    def test_generate_includes_all_stages(self, sample_topic):
        """Test that generated PRD includes all stages."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        stages = {story.stage for story in prd.user_stories}
        expected_stages = {
            "literature", "hypothesis", "data", "experiment",
            "analysis", "visualization", "writing", "review"
        }

        assert stages == expected_stages

    def test_generate_has_correct_story_count(self, sample_topic):
        """Test that generated PRD has expected number of stories."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        expected_count = (
            len(LITERATURE_STORIES) +
            len(HYPOTHESIS_STORIES) +
            len(DATA_STORIES) +
            len(EXPERIMENT_STORIES) +
            len(ANALYSIS_STORIES) +
            len(VISUALIZATION_STORIES) +
            len(WRITING_STORIES) +
            len(REVIEW_STORIES)
        )

        assert len(prd.user_stories) == expected_count

    def test_generate_creates_valid_branch_name(self, sample_topic):
        """Test that branch name is created correctly."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("My Test Project")

        assert prd.branch_name == "my-test-project"
        assert " " not in prd.branch_name

    def test_generate_long_project_name_truncated(self, sample_topic):
        """Test that long project names are truncated in branch name."""
        generator = PRDGenerator(sample_topic)
        long_name = "a" * 100
        prd = generator.generate(long_name)

        assert len(prd.branch_name) <= 50

    def test_generate_adds_domain_skills(self, sample_topic):
        """Test that domain-specific skills are added."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        # NLP domain should have NLP-specific skills
        exp_stories = [s for s in prd.user_stories if s.stage == "experiment"]

        # At least one experiment story should have NLP skills
        nlp_skills = {"transformers", "nltk", "spacy"}
        has_nlp_skill = any(
            nlp_skills & set(story.skills_required)
            for story in exp_stories
        )

        assert has_nlp_skill

    def test_generate_respects_gpu_constraint(self, sample_topic):
        """Test that GPU constraint is noted in stories."""
        # Create topic without GPU requirement
        no_gpu_topic = ResearchTopic(
            title="Simple ML Research Project Title",
            hypothesis="A hypothesis that is long enough to pass the minimum length validation requirements.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")],
            constraints=Constraints(gpu_required=False)
        )

        generator = PRDGenerator(no_gpu_topic)
        prd = generator.generate("test-project")

        # Stories with pytorch should note CPU-only
        pytorch_stories = [
            s for s in prd.user_stories
            if "pytorch" in s.skills_required
        ]

        for story in pytorch_stories:
            assert "CPU" in story.notes or story.notes == ""

    def test_stories_have_valid_dependencies(self, sample_topic):
        """Test that all story dependencies are valid."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        story_ids = {story.id for story in prd.user_stories}

        for story in prd.user_stories:
            for dep in story.dependencies:
                assert dep in story_ids, f"Story {story.id} has invalid dependency: {dep}"

    def test_stories_have_unique_ids(self, sample_topic):
        """Test that all story IDs are unique."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test-project")

        ids = [story.id for story in prd.user_stories]
        assert len(ids) == len(set(ids))


class TestGeneratePRDFunction:
    """Tests for generate_prd convenience function."""

    def test_generate_prd_returns_prd(self, sample_topic):
        """Test that generate_prd returns a PRD."""
        prd = generate_prd("test-project", sample_topic)

        assert prd is not None
        assert prd.project == "test-project"

    def test_generate_prd_saves_to_file(self, sample_topic, tmp_path):
        """Test that generate_prd can save to file."""
        output_path = tmp_path / "test_prd.json"

        prd = generate_prd("test-project", sample_topic, str(output_path))

        assert output_path.exists()

        # Verify file content
        loaded_prd = prd.from_file(output_path)
        assert loaded_prd.project == prd.project


class TestDomainSpecificSkills:
    """Tests for domain-specific skill assignment."""

    def test_ml_domain_skills(self):
        """Test ML domain adds correct skills."""
        topic = ResearchTopic(
            title="Machine Learning Research Project",
            hypothesis="A hypothesis that is long enough to pass the minimum length validation requirements.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        generator = PRDGenerator(topic)
        prd = generator.generate("test")

        all_skills = set()
        for story in prd.user_stories:
            all_skills.update(story.skills_required)

        assert "scikit-learn" in all_skills or "pytorch" in all_skills

    def test_bioinformatics_domain_skills(self):
        """Test bioinformatics domain adds correct skills."""
        topic = ResearchTopic(
            title="Bioinformatics Research Project Title",
            hypothesis="A hypothesis that is long enough to pass the minimum length validation requirements.",
            domain="BIOINFORMATICS",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        generator = PRDGenerator(topic)
        prd = generator.generate("test")

        all_skills = set()
        for story in prd.user_stories:
            all_skills.update(story.skills_required)

        assert "bioinformatics" in all_skills or "biopython" in all_skills

    def test_statistics_domain_skills(self):
        """Test statistics domain adds correct skills."""
        topic = ResearchTopic(
            title="Statistical Research Project Title Here",
            hypothesis="A hypothesis that is long enough to pass the minimum length validation requirements.",
            domain="STATISTICS",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        generator = PRDGenerator(topic)
        prd = generator.generate("test")

        all_skills = set()
        for story in prd.user_stories:
            all_skills.update(story.skills_required)

        assert "statistical-analysis" in all_skills or "scipy" in all_skills

    def test_computer_vision_domain_skills(self):
        """Test computer vision domain adds correct skills."""
        topic = ResearchTopic(
            title="Computer Vision Research Project Title",
            hypothesis="A hypothesis that is long enough to pass the minimum length validation requirements.",
            domain="COMPUTER_VISION",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        generator = PRDGenerator(topic)
        prd = generator.generate("test")

        all_skills = set()
        for story in prd.user_stories:
            all_skills.update(story.skills_required)

        assert "pytorch" in all_skills or "torchvision" in all_skills or "opencv" in all_skills


class TestStoryOrdering:
    """Tests for story ordering and dependencies."""

    def test_literature_comes_first(self, sample_topic):
        """Test that literature stories have lowest priorities."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test")

        lit_stories = [s for s in prd.user_stories if s.stage == "literature"]
        other_stories = [s for s in prd.user_stories if s.stage != "literature"]

        # First ready story should be from literature
        ready = prd.get_ready_stories()
        assert ready[0].stage == "literature"

    def test_review_comes_last(self, sample_topic):
        """Test that review stories come after writing."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test")

        review_stories = [s for s in prd.user_stories if s.stage == "review"]

        for story in review_stories:
            for dep in story.dependencies:
                dep_story = next(s for s in prd.user_stories if s.id == dep)
                # Dependencies should be from writing or earlier stages
                assert dep_story.stage in ["writing", "visualization", "analysis", "experiment"]

    def test_no_circular_dependencies(self, sample_topic):
        """Test that there are no circular dependencies."""
        generator = PRDGenerator(sample_topic)
        prd = generator.generate("test")

        # This is validated by Pydantic, but let's verify
        # Build dependency graph
        graph = {s.id: set(s.dependencies) for s in prd.user_stories}

        # Topological sort should work without error
        visited = set()
        temp_visited = set()

        def visit(node):
            if node in temp_visited:
                raise ValueError("Circular dependency detected")
            if node in visited:
                return
            temp_visited.add(node)
            for dep in graph.get(node, []):
                visit(dep)
            temp_visited.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)

        # If we get here, no circular dependencies
        assert True
