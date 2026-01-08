"""Unit tests for quality gates."""

import json
import tempfile
from pathlib import Path

import pytest

from lib.db.manager import DatabaseManager
from lib.quality_gates import (
    DataGate,
    ExperimentGate,
    FinalGate,
    GateCheckResult,
    GateManager,
    LiteratureGate,
    QualityGate,
    StatisticsGate,
    WritingGate,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create minimal schema
    schema = """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS research_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL UNIQUE,
            topic_json TEXT NOT NULL,
            status TEXT DEFAULT 'initialized',
            current_stage TEXT,
            iteration_count INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS literature (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            relevance_score REAL,
            included_in_review BOOLEAN DEFAULT FALSE,
            screening_status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            source TEXT NOT NULL,
            identifier TEXT NOT NULL,
            downloaded BOOLEAN DEFAULT FALSE,
            eda_completed BOOLEAN DEFAULT FALSE,
            preprocessing_completed BOOLEAN DEFAULT FALSE,
            train_size REAL,
            val_size REAL,
            test_size REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            experiment_name TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            metrics_json TEXT,
            random_seed INTEGER,
            model_path TEXT,
            notebook_path TEXT,
            runtime_seconds REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        );

        CREATE TABLE IF NOT EXISTS hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            hypothesis_text TEXT NOT NULL,
            source TEXT NOT NULL,
            tested BOOLEAN DEFAULT FALSE,
            priority INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS statistical_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            test_name TEXT NOT NULL,
            effect_size REAL,
            ci_lower REAL,
            ci_upper REAL,
            assumptions_checked BOOLEAN DEFAULT FALSE,
            significant BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS paper_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            section_name TEXT NOT NULL,
            version INTEGER DEFAULT 1,
            content TEXT,
            word_count INTEGER,
            status TEXT DEFAULT 'outline',
            section_order INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id, section_name, version)
        );

        CREATE TABLE IF NOT EXISTS figures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            figure_number INTEGER,
            figure_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            included_in_paper BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            iteration INTEGER NOT NULL,
            story_id TEXT,
            action TEXT NOT NULL,
            action_type TEXT,
            result TEXT,
            output_summary TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.close()

    db = DatabaseManager(db_path)
    yield db

    db_path.unlink(missing_ok=True)


@pytest.fixture
def project_id(temp_db):
    """Create a test project."""
    return temp_db.create_project("test-project", json.dumps({"title": "Test"}))


class TestGateCheckResult:
    """Tests for GateCheckResult dataclass."""

    def test_create_passed_result(self):
        """Test creating a passed result."""
        result = GateCheckResult(
            check_name="test_check",
            passed=True,
            score=0.9,
            message="Check passed"
        )
        assert result.passed is True
        assert result.score == 0.9

    def test_create_failed_result_with_details(self):
        """Test creating a failed result with details."""
        result = GateCheckResult(
            check_name="test_check",
            passed=False,
            score=0.3,
            message="Check failed",
            details={"reason": "Not enough data"}
        )
        assert result.passed is False
        assert result.details["reason"] == "Not enough data"


class TestLiteratureGate:
    """Tests for LiteratureGate."""

    def test_fails_with_no_literature(self, temp_db, project_id):
        """Test that gate fails with no literature."""
        gate = LiteratureGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False
        assert result.score < gate.threshold

    def test_fails_with_insufficient_papers(self, temp_db, project_id):
        """Test that gate fails with too few papers."""
        # Add only 5 papers (need 20)
        for i in range(5):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex"
            )

        gate = LiteratureGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_fails_with_insufficient_included(self, temp_db, project_id):
        """Test that gate fails with too few included papers."""
        # Add 25 papers but only include 5
        for i in range(25):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex",
                relevance_score=0.5,
                included_in_review=(i < 5)
            )

        gate = LiteratureGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_passes_with_sufficient_literature(self, temp_db, project_id):
        """Test that gate passes with sufficient literature."""
        # Add 25 papers with 15 included
        for i in range(25):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex" if i % 2 == 0 else "semantic_scholar",
                relevance_score=0.8,
                included_in_review=(i < 15)
            )

        gate = LiteratureGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is True
        assert result.score >= gate.threshold

    def test_checks_multiple_sources(self, temp_db, project_id):
        """Test that gate checks for multiple sources."""
        # Add papers from only one source
        for i in range(30):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex",  # Single source
                relevance_score=0.8,
                included_in_review=(i < 15)
            )

        gate = LiteratureGate()
        result = gate.evaluate(project_id, temp_db)

        # Should fail or have lower score due to single source
        checks = result.details.get("checks", [])
        source_check = next(
            (c for c in checks if c["name"] == "multiple_sources_used"),
            None
        )
        assert source_check is not None
        assert source_check["passed"] is False


class TestDataGate:
    """Tests for DataGate."""

    def test_fails_with_no_datasets(self, temp_db, project_id):
        """Test that gate fails with no datasets."""
        gate = DataGate()
        result = gate.evaluate(project_id, temp_db)

        # With no datasets, the gate should fail
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_with_incomplete_dataset(self, temp_db, project_id):
        """Test that gate fails with incomplete dataset processing."""
        temp_db.insert_dataset(
            project_id,
            source="kaggle",
            identifier="test/data",
            downloaded=True,
            eda_completed=False,  # Not done
            preprocessing_completed=False
        )

        gate = DataGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_passes_with_complete_dataset(self, temp_db, project_id):
        """Test that gate passes with complete dataset."""
        temp_db.insert_dataset(
            project_id,
            source="kaggle",
            identifier="test/data",
            downloaded=True,
            eda_completed=True,
            preprocessing_completed=True,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )

        gate = DataGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is True


class TestExperimentGate:
    """Tests for ExperimentGate."""

    def test_fails_with_no_experiments(self, temp_db, project_id):
        """Test that gate fails with no experiments."""
        gate = ExperimentGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_fails_with_insufficient_baselines(self, temp_db, project_id):
        """Test that gate fails with too few baselines."""
        # Add only 1 baseline (need 2)
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="baseline1",
            experiment_type="baseline",
            random_seed=42
        )
        temp_db.update_experiment_results(exp_id, json.dumps({"acc": 0.8}), 10.0)

        gate = ExperimentGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_fails_without_proposed_method(self, temp_db, project_id):
        """Test that gate fails without proposed method."""
        # Add 2 baselines but no proposed
        for i in range(2):
            exp_id = temp_db.insert_experiment(
                project_id,
                experiment_name=f"baseline{i}",
                experiment_type="baseline",
                random_seed=42
            )
            temp_db.update_experiment_results(exp_id, json.dumps({"acc": 0.8}), 10.0)

        # With threshold at 0.76, missing proposed method (score 0.75) should fail
        gate = ExperimentGate(threshold=0.76)
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_passes_with_complete_experiments(self, temp_db, project_id):
        """Test that gate passes with complete experiments."""
        # Add 2 baselines
        for i in range(2):
            exp_id = temp_db.insert_experiment(
                project_id,
                experiment_name=f"baseline{i}",
                experiment_type="baseline",
                random_seed=42
            )
            temp_db.update_experiment_results(exp_id, json.dumps({"acc": 0.8}), 10.0)

        # Add proposed method
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="proposed",
            experiment_type="proposed",
            random_seed=42
        )
        temp_db.update_experiment_results(exp_id, json.dumps({"acc": 0.9}), 20.0)

        gate = ExperimentGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is True

    def test_checks_reproducibility_seeds(self, temp_db, project_id):
        """Test that gate checks for reproducibility seeds."""
        # Add experiment without seed
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="baseline",
            experiment_type="baseline"
            # No random_seed
        )
        temp_db.update("experiments", {"status": "completed", "metrics_json": "{}"}, "id = ?", (exp_id,))

        gate = ExperimentGate()
        result = gate.evaluate(project_id, temp_db)

        checks = result.details.get("checks", [])
        seed_check = next(
            (c for c in checks if c["name"] == "reproducibility_seeds_set"),
            None
        )
        assert seed_check is not None


class TestStatisticsGate:
    """Tests for StatisticsGate."""

    def test_fails_with_no_tests(self, temp_db, project_id):
        """Test that gate fails with no statistical tests."""
        gate = StatisticsGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_passes_with_complete_statistics(self, temp_db, project_id):
        """Test that gate passes with complete statistical analysis."""
        temp_db.insert_statistical_test(
            project_id,
            test_name="paired_t_test",
            effect_size=0.5,
            ci_lower=0.1,
            ci_upper=0.9,
            assumptions_checked=True,
            significant=True
        )

        gate = StatisticsGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is True


class TestWritingGate:
    """Tests for WritingGate."""

    def test_fails_with_no_sections(self, temp_db, project_id):
        """Test that gate fails with no paper sections."""
        gate = WritingGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_fails_with_missing_sections(self, temp_db, project_id):
        """Test that gate fails with missing required sections."""
        # Add only abstract
        temp_db.insert_paper_section(
            project_id,
            section_name="abstract",
            content="This is the abstract.",
            word_count=150,
            status="draft"
        )

        gate = WritingGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_passes_with_all_sections(self, temp_db, project_id):
        """Test that gate passes with all required sections."""
        required_sections = [
            "abstract", "introduction", "related_work",
            "methodology", "experiments", "results", "conclusion"
        ]

        for i, section in enumerate(required_sections):
            word_count = 200 if section == "abstract" else 500
            temp_db.insert_paper_section(
                project_id,
                section_name=section,
                content=f"Content for {section}",
                word_count=word_count,
                status="draft",
                section_order=i
            )

        # Add some figures
        temp_db.insert_figure(
            project_id,
            figure_type="results",
            file_path="figures/results.pdf",
            included_in_paper=True
        )
        temp_db.insert_figure(
            project_id,
            figure_type="comparison",
            file_path="figures/comparison.pdf",
            included_in_paper=True
        )

        gate = WritingGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is True


class TestFinalGate:
    """Tests for FinalGate."""

    def test_fails_without_prior_gates(self, temp_db, project_id):
        """Test that final gate fails without prior gates passing."""
        gate = FinalGate()
        result = gate.evaluate(project_id, temp_db)

        assert result.passed is False

    def test_checks_hypotheses_tested(self, temp_db, project_id):
        """Test that final gate checks for tested hypotheses."""
        temp_db.insert("hypotheses", {
            "project_id": project_id,
            "hypothesis_text": "Test hypothesis",
            "source": "initial",
            "tested": True
        })

        gate = FinalGate()
        result = gate.evaluate(project_id, temp_db)

        checks = result.details.get("checks", [])
        hyp_check = next(
            (c for c in checks if c["name"] == "hypotheses_tested"),
            None
        )
        assert hyp_check is not None
        assert hyp_check["passed"] is True


class TestQualityGateBase:
    """Tests for base QualityGate class."""

    def test_can_retry_initially_true(self, temp_db, project_id):
        """Test that can_retry is initially True."""
        gate = LiteratureGate()
        assert gate.can_retry() is True

    def test_can_retry_after_max_attempts(self, temp_db, project_id):
        """Test that can_retry becomes False after max attempts."""
        gate = LiteratureGate()
        gate.max_retries = 2

        gate.increment_retry()
        assert gate.can_retry() is True

        gate.increment_retry()
        assert gate.can_retry() is False

    def test_threshold_is_used(self, temp_db, project_id):
        """Test that threshold is correctly used."""
        gate = LiteratureGate(threshold=0.9)

        assert gate.threshold == 0.9


class TestGateManager:
    """Tests for GateManager."""

    def test_get_gate(self, temp_db):
        """Test getting a gate by stage."""
        manager = GateManager(temp_db)

        lit_gate = manager.get_gate("literature")
        assert lit_gate is not None
        assert isinstance(lit_gate, LiteratureGate)

        data_gate = manager.get_gate("data")
        assert data_gate is not None
        assert isinstance(data_gate, DataGate)

    def test_get_unknown_gate(self, temp_db):
        """Test getting unknown gate returns None."""
        manager = GateManager(temp_db)

        gate = manager.get_gate("unknown")
        assert gate is None

    def test_evaluate_stage(self, temp_db, project_id):
        """Test evaluating a stage gate."""
        manager = GateManager(temp_db)

        result = manager.evaluate_stage(project_id, "literature")

        assert result.gate_name == "LiteratureGate"
        assert "passed" in dir(result)

    def test_evaluate_unknown_stage(self, temp_db, project_id):
        """Test evaluating unknown stage returns passed result."""
        manager = GateManager(temp_db)

        result = manager.evaluate_stage(project_id, "unknown")

        # Unknown stages should pass by default
        assert result.passed is True

    def test_evaluate_all(self, temp_db, project_id):
        """Test evaluating all gates."""
        manager = GateManager(temp_db)

        results = manager.evaluate_all(project_id)

        assert "literature" in results
        assert "data" in results
        assert "experiment" in results
        assert "analysis" in results
        assert "writing" in results
        assert "final" in results

    def test_can_proceed(self, temp_db, project_id):
        """Test can_proceed check."""
        # Set up passing literature gate
        for i in range(30):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex" if i % 2 == 0 else "semantic_scholar",
                relevance_score=0.8,
                included_in_review=(i < 15)
            )

        manager = GateManager(temp_db)

        can_proceed = manager.can_proceed(project_id, "literature", "data")

        assert isinstance(can_proceed, bool)
