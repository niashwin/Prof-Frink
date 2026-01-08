"""Unit tests for DatabaseManager."""

import json
import tempfile
from pathlib import Path

import pytest

from homer.lib.db.manager import DatabaseManager, get_database_manager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Copy schema to temp location
    schema_path = Path(__file__).parent.parent.parent / "lib" / "db" / "schema.sql"
    if not schema_path.exists():
        # Create minimal schema for testing
        temp_schema = db_path.parent / "schema.sql"
        temp_schema.write_text("""
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS research_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL UNIQUE,
                topic_json TEXT NOT NULL,
                status TEXT DEFAULT 'initialized',
                current_stage TEXT,
                current_story_id TEXT,
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
                exclusion_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES research_projects(id),
                hypothesis_text TEXT NOT NULL,
                source TEXT NOT NULL,
                tested BOOLEAN DEFAULT FALSE,
                result TEXT,
                evidence_summary TEXT,
                supporting_experiments TEXT,
                tested_at DATETIME,
                priority INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES research_projects(id),
                source TEXT NOT NULL,
                identifier TEXT NOT NULL,
                name TEXT,
                downloaded BOOLEAN DEFAULT FALSE,
                eda_completed BOOLEAN DEFAULT FALSE,
                preprocessing_completed BOOLEAN DEFAULT FALSE,
                target_column TEXT,
                feature_columns TEXT,
                categorical_columns TEXT,
                numerical_columns TEXT,
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
                runtime_seconds REAL,
                memory_mb REAL,
                model_path TEXT,
                predictions_path TEXT,
                notebook_path TEXT,
                error_message TEXT,
                started_at DATETIME,
                completed_at DATETIME,
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
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, section_name, version)
            );

            CREATE TABLE IF NOT EXISTS figures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES research_projects(id),
                figure_type TEXT NOT NULL,
                figure_number INTEGER,
                file_path TEXT NOT NULL,
                included_in_paper BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES research_projects(id),
                table_type TEXT NOT NULL,
                table_number INTEGER,
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
                skill_used TEXT,
                skill_params_json TEXT,
                result TEXT,
                output_summary TEXT,
                error_message TEXT,
                learnings TEXT,
                patterns_discovered TEXT,
                duration_seconds REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES research_projects(id),
                checkpoint_name TEXT NOT NULL,
                checkpoint_type TEXT DEFAULT 'automatic',
                prd_json TEXT NOT NULL,
                progress_txt TEXT,
                git_commit_hash TEXT,
                stage_completed TEXT,
                stories_completed INTEGER,
                stories_total INTEGER,
                trigger_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, checkpoint_name)
            );

            CREATE VIEW IF NOT EXISTS v_project_summary AS
            SELECT
                p.id,
                p.project_name,
                p.status,
                p.current_stage,
                p.iteration_count,
                (SELECT COUNT(*) FROM literature WHERE project_id = p.id AND included_in_review = 1) as papers_included,
                (SELECT COUNT(*) FROM experiments WHERE project_id = p.id AND status = 'completed') as experiments_completed,
                (SELECT COUNT(*) FROM statistical_tests WHERE project_id = p.id) as tests_performed,
                (SELECT COUNT(*) FROM figures WHERE project_id = p.id AND included_in_paper = 1) as figures_included,
                (SELECT COUNT(*) FROM paper_sections WHERE project_id = p.id AND status = 'final') as sections_final,
                p.created_at,
                p.updated_at
            FROM research_projects p;

            CREATE VIEW IF NOT EXISTS v_experiment_results AS
            SELECT
                e.id,
                e.project_id,
                e.experiment_name,
                e.experiment_type,
                e.status,
                e.metrics_json,
                e.runtime_seconds
            FROM experiments e;
        """)

    db = DatabaseManager(db_path)
    yield db

    # Cleanup
    db_path.unlink(missing_ok=True)


class TestDatabaseManagerInit:
    """Tests for DatabaseManager initialization."""

    def test_creates_database_file(self, temp_db):
        """Test that database file is created."""
        assert temp_db.db_path.exists()

    def test_connection_context_manager(self, temp_db):
        """Test connection context manager works."""
        with temp_db.connection() as conn:
            assert conn is not None
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

    def test_get_database_manager_function(self):
        """Test get_database_manager convenience function."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            db = get_database_manager(db_path)
            assert db.db_path == db_path
        finally:
            db_path.unlink(missing_ok=True)


class TestProjectOperations:
    """Tests for project CRUD operations."""

    def test_create_project(self, temp_db):
        """Test creating a new project."""
        topic_json = json.dumps({"title": "Test Research"})
        project_id = temp_db.create_project("test-project", topic_json)

        assert project_id is not None
        assert project_id > 0

    def test_get_project(self, temp_db):
        """Test retrieving a project by ID."""
        topic_json = json.dumps({"title": "Test Research"})
        project_id = temp_db.create_project("test-project", topic_json)

        project = temp_db.get_project(project_id)

        assert project is not None
        assert project["project_name"] == "test-project"
        assert project["status"] == "initialized"

    def test_get_project_by_name(self, temp_db):
        """Test retrieving a project by name."""
        topic_json = json.dumps({"title": "Test Research"})
        temp_db.create_project("unique-name", topic_json)

        project = temp_db.get_project_by_name("unique-name")

        assert project is not None
        assert project["project_name"] == "unique-name"

    def test_get_nonexistent_project(self, temp_db):
        """Test retrieving a project that doesn't exist."""
        project = temp_db.get_project(99999)
        assert project is None

    def test_update_project_status(self, temp_db):
        """Test updating project status."""
        topic_json = json.dumps({"title": "Test"})
        project_id = temp_db.create_project("test", topic_json)

        temp_db.update_project_status(project_id, "in_progress", "literature")

        project = temp_db.get_project(project_id)
        assert project["status"] == "in_progress"
        assert project["current_stage"] == "literature"

    def test_increment_iteration(self, temp_db):
        """Test incrementing iteration count."""
        topic_json = json.dumps({"title": "Test"})
        project_id = temp_db.create_project("test", topic_json)

        new_count = temp_db.increment_iteration(project_id)
        assert new_count == 1

        new_count = temp_db.increment_iteration(project_id)
        assert new_count == 2

    def test_list_projects(self, temp_db):
        """Test listing all projects."""
        temp_db.create_project("project1", json.dumps({"title": "P1"}))
        temp_db.create_project("project2", json.dumps({"title": "P2"}))

        projects = temp_db.list_projects()

        assert len(projects) == 2

    def test_list_projects_by_status(self, temp_db):
        """Test listing projects filtered by status."""
        p1 = temp_db.create_project("project1", json.dumps({"title": "P1"}))
        temp_db.create_project("project2", json.dumps({"title": "P2"}))
        temp_db.update_project_status(p1, "completed")

        completed = temp_db.list_projects("completed")

        assert len(completed) == 1
        assert completed[0]["project_name"] == "project1"


class TestLiteratureOperations:
    """Tests for literature CRUD operations."""

    @pytest.fixture
    def project_id(self, temp_db):
        """Create a project for literature tests."""
        return temp_db.create_project("lit-test", json.dumps({"title": "Test"}))

    def test_insert_literature(self, temp_db, project_id):
        """Test inserting literature entry."""
        lit_id = temp_db.insert_literature(
            project_id,
            title="Test Paper",
            source="openalex",
            relevance_score=0.85
        )

        assert lit_id is not None
        assert lit_id > 0

    def test_get_included_literature(self, temp_db, project_id):
        """Test getting included literature."""
        temp_db.insert_literature(
            project_id,
            title="Included Paper",
            source="openalex",
            relevance_score=0.9,
            included_in_review=True
        )
        temp_db.insert_literature(
            project_id,
            title="Excluded Paper",
            source="openalex",
            relevance_score=0.3,
            included_in_review=False
        )

        included = temp_db.get_included_literature(project_id)

        assert len(included) == 1
        assert included[0]["title"] == "Included Paper"

    def test_count_literature(self, temp_db, project_id):
        """Test counting literature entries."""
        temp_db.insert_literature(project_id, title="P1", source="openalex", included_in_review=True)
        temp_db.insert_literature(project_id, title="P2", source="openalex", included_in_review=True)
        temp_db.insert_literature(project_id, title="P3", source="openalex", included_in_review=False)

        total = temp_db.count_literature(project_id)
        included = temp_db.count_literature(project_id, included_only=True)

        assert total == 3
        assert included == 2


class TestExperimentOperations:
    """Tests for experiment CRUD operations."""

    @pytest.fixture
    def project_id(self, temp_db):
        """Create a project for experiment tests."""
        return temp_db.create_project("exp-test", json.dumps({"title": "Test"}))

    def test_insert_experiment(self, temp_db, project_id):
        """Test inserting an experiment."""
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="baseline_rf",
            experiment_type="baseline",
            random_seed=42
        )

        assert exp_id is not None
        assert exp_id > 0

    def test_get_experiments_by_type(self, temp_db, project_id):
        """Test getting experiments by type."""
        temp_db.insert_experiment(project_id, experiment_name="baseline1", experiment_type="baseline")
        temp_db.insert_experiment(project_id, experiment_name="baseline2", experiment_type="baseline")
        temp_db.insert_experiment(project_id, experiment_name="proposed", experiment_type="proposed")

        baselines = temp_db.get_experiments_by_type(project_id, "baseline")

        assert len(baselines) == 2

    def test_update_experiment_status(self, temp_db, project_id):
        """Test updating experiment status."""
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="test",
            experiment_type="baseline"
        )

        temp_db.update_experiment_status(exp_id, "running")
        exp = temp_db.get_experiment(exp_id)
        assert exp["status"] == "running"
        assert exp["started_at"] is not None

    def test_update_experiment_results(self, temp_db, project_id):
        """Test updating experiment with results."""
        exp_id = temp_db.insert_experiment(
            project_id,
            experiment_name="test",
            experiment_type="baseline"
        )

        metrics = json.dumps({"accuracy": 0.95, "f1": 0.92})
        temp_db.update_experiment_results(
            exp_id,
            metrics_json=metrics,
            runtime_seconds=120.5,
            memory_mb=512.0
        )

        exp = temp_db.get_experiment(exp_id)
        assert exp["status"] == "completed"
        assert exp["metrics_json"] == metrics
        assert exp["runtime_seconds"] == 120.5

    def test_get_completed_experiments(self, temp_db, project_id):
        """Test getting completed experiments."""
        exp1 = temp_db.insert_experiment(project_id, experiment_name="exp1", experiment_type="baseline")
        exp2 = temp_db.insert_experiment(project_id, experiment_name="exp2", experiment_type="baseline")

        temp_db.update_experiment_results(exp1, json.dumps({}), 10.0)

        completed = temp_db.get_completed_experiments(project_id)

        assert len(completed) == 1
        assert completed[0]["experiment_name"] == "exp1"


class TestCheckpointOperations:
    """Tests for checkpoint operations."""

    @pytest.fixture
    def project_id(self, temp_db):
        """Create a project for checkpoint tests."""
        return temp_db.create_project("cp-test", json.dumps({"title": "Test"}))

    def test_create_checkpoint(self, temp_db, project_id):
        """Test creating a checkpoint."""
        cp_id = temp_db.create_checkpoint(
            project_id,
            name="test-checkpoint",
            prd_json='{"project": "test"}',
            progress_txt="Some progress",
            git_hash="abc123"
        )

        assert cp_id is not None
        assert cp_id > 0

    def test_get_latest_checkpoint(self, temp_db, project_id):
        """Test getting latest checkpoint."""
        temp_db.create_checkpoint(project_id, name="cp1", prd_json="{}", progress_txt="")
        temp_db.create_checkpoint(project_id, name="cp2", prd_json="{}", progress_txt="")

        latest = temp_db.get_latest_checkpoint(project_id)

        assert latest is not None
        assert latest["checkpoint_name"] == "cp2"

    def test_get_checkpoint_by_name(self, temp_db, project_id):
        """Test getting checkpoint by name."""
        temp_db.create_checkpoint(project_id, name="specific-cp", prd_json="{}", progress_txt="")

        cp = temp_db.get_checkpoint_by_name(project_id, "specific-cp")

        assert cp is not None
        assert cp["checkpoint_name"] == "specific-cp"

    def test_list_checkpoints(self, temp_db, project_id):
        """Test listing all checkpoints."""
        temp_db.create_checkpoint(project_id, name="cp1", prd_json="{}", progress_txt="")
        temp_db.create_checkpoint(project_id, name="cp2", prd_json="{}", progress_txt="")

        checkpoints = temp_db.list_checkpoints(project_id)

        assert len(checkpoints) == 2


class TestAgentLogOperations:
    """Tests for agent log operations."""

    @pytest.fixture
    def project_id(self, temp_db):
        """Create a project for log tests."""
        return temp_db.create_project("log-test", json.dumps({"title": "Test"}))

    def test_log_agent_action(self, temp_db, project_id):
        """Test logging an agent action."""
        log_id = temp_db.log_agent_action(
            project_id,
            iteration=1,
            story_id="LIT-001",
            action="Starting literature search",
            action_type="story_start"
        )

        assert log_id is not None
        assert log_id > 0

    def test_get_agent_log(self, temp_db, project_id):
        """Test getting agent log."""
        temp_db.log_agent_action(project_id, 1, "LIT-001", "Action 1", "story_start")
        temp_db.log_agent_action(project_id, 2, "LIT-002", "Action 2", "story_start")

        logs = temp_db.get_agent_log(project_id)

        assert len(logs) == 2

    def test_get_agent_log_by_iteration(self, temp_db, project_id):
        """Test getting log filtered by iteration."""
        temp_db.log_agent_action(project_id, 1, "LIT-001", "Action 1", "story_start")
        temp_db.log_agent_action(project_id, 2, "LIT-002", "Action 2", "story_start")

        logs = temp_db.get_agent_log(project_id, iteration=1)

        assert len(logs) == 1
        assert logs[0]["iteration"] == 1

    def test_get_learnings(self, temp_db, project_id):
        """Test getting logged learnings."""
        temp_db.log_agent_action(
            project_id, 1, "LIT-001", "Action 1", "complete",
            learnings="Important finding"
        )
        temp_db.log_agent_action(
            project_id, 2, "LIT-002", "Action 2", "complete"
        )

        learnings = temp_db.get_learnings(project_id)

        assert len(learnings) == 1
        assert "Important finding" in learnings[0]["learnings"]


class TestGenericOperations:
    """Tests for generic database operations."""

    def test_insert_and_fetch_one(self, temp_db):
        """Test generic insert and fetch_one."""
        topic_json = json.dumps({"title": "Test"})
        project_id = temp_db.insert("research_projects", {
            "project_name": "generic-test",
            "topic_json": topic_json
        })

        result = temp_db.fetch_one(
            "SELECT * FROM research_projects WHERE id = ?",
            (project_id,)
        )

        assert result is not None
        assert result["project_name"] == "generic-test"

    def test_update(self, temp_db):
        """Test generic update."""
        topic_json = json.dumps({"title": "Test"})
        project_id = temp_db.create_project("update-test", topic_json)

        affected = temp_db.update(
            "research_projects",
            {"status": "completed"},
            "id = ?",
            (project_id,)
        )

        assert affected == 1

        project = temp_db.get_project(project_id)
        assert project["status"] == "completed"

    def test_delete(self, temp_db):
        """Test generic delete."""
        topic_json = json.dumps({"title": "Test"})
        project_id = temp_db.create_project("delete-test", topic_json)

        affected = temp_db.delete("research_projects", "id = ?", (project_id,))

        assert affected == 1
        assert temp_db.get_project(project_id) is None

    def test_fetch_all(self, temp_db):
        """Test generic fetch_all."""
        temp_db.create_project("p1", json.dumps({"title": "P1"}))
        temp_db.create_project("p2", json.dumps({"title": "P2"}))

        results = temp_db.fetch_all("SELECT * FROM research_projects")

        assert len(results) == 2

    def test_count(self, temp_db):
        """Test generic count."""
        temp_db.create_project("p1", json.dumps({"title": "P1"}))
        temp_db.create_project("p2", json.dumps({"title": "P2"}))

        count = temp_db.count("SELECT COUNT(*) FROM research_projects")

        assert count == 2

    def test_transaction_rollback_on_error(self, temp_db):
        """Test that transactions roll back on error."""
        temp_db.create_project("rollback-test", json.dumps({"title": "Test"}))

        # Try to insert duplicate (should fail)
        with pytest.raises(Exception):
            temp_db.create_project("rollback-test", json.dumps({"title": "Test"}))

        # Original should still exist
        project = temp_db.get_project_by_name("rollback-test")
        assert project is not None


class TestViewQueries:
    """Tests for view queries."""

    def test_get_project_summary(self, temp_db):
        """Test project summary view."""
        project_id = temp_db.create_project("summary-test", json.dumps({"title": "Test"}))

        summary = temp_db.get_project_summary(project_id)

        assert summary is not None
        assert summary["project_name"] == "summary-test"
        assert "papers_included" in summary
        assert "experiments_completed" in summary

    def test_get_experiment_results_view(self, temp_db):
        """Test experiment results view."""
        project_id = temp_db.create_project("results-test", json.dumps({"title": "Test"}))
        temp_db.insert_experiment(project_id, experiment_name="exp1", experiment_type="baseline")

        results = temp_db.get_experiment_results_view(project_id)

        assert len(results) == 1
        assert results[0]["experiment_name"] == "exp1"
