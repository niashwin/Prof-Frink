"""Pytest configuration and shared fixtures for FRINK tests."""

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# =============================================================================
# DATABASE FIXTURES
# =============================================================================


@pytest.fixture(scope="function")
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture(scope="function")
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def minimal_db_schema() -> str:
    """Return minimal database schema for testing."""
    return """
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
    """


# =============================================================================
# TOPIC FIXTURES
# =============================================================================


@pytest.fixture
def sample_topic_data() -> dict:
    """Return sample topic data for testing."""
    return {
        "title": "Investigating Machine Learning Methods for Classification",
        "hypothesis": "Ensemble methods will outperform single models for binary classification tasks due to their ability to reduce variance and combine diverse learning perspectives.",
        "domain": "ML",
        "datasets": [
            {
                "source": "kaggle",
                "identifier": "test/sample-dataset",
                "description": "A sample dataset for testing"
            }
        ],
        "research_questions": [
            "Which algorithm performs best?",
            "How does feature selection affect performance?"
        ],
        "constraints": {
            "max_compute_hours": 4.0,
            "gpu_required": False
        },
        "baseline_requirements": {
            "minimum_baselines": 2,
            "include_sota": True
        },
        "keywords": ["machine learning", "classification", "ensemble"]
    }


@pytest.fixture
def sample_prd_data(sample_topic_data: dict) -> dict:
    """Return sample PRD data for testing."""
    return {
        "project": "test-project",
        "branchName": "test-project",
        "topic": sample_topic_data,
        "userStories": [
            {
                "id": "LIT-001",
                "title": "Search academic databases",
                "stage": "literature",
                "priority": 1
            },
            {
                "id": "LIT-002",
                "title": "Screen papers",
                "stage": "literature",
                "priority": 2,
                "dependencies": ["LIT-001"]
            },
            {
                "id": "DATA-001",
                "title": "Download dataset",
                "stage": "data",
                "priority": 3,
                "dependencies": ["LIT-002"]
            }
        ]
    }


# =============================================================================
# MARKERS AND CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("-m"):
        return  # Don't modify if markers are explicitly specified

    # Add slow marker to e2e tests
    for item in items:
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.e2e)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_test_database(db_path: Path, schema: str) -> None:
    """Create a test database with the given schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.close()


def seed_literature(db, project_id: int, count: int = 20) -> None:
    """Seed literature data for testing."""
    for i in range(count):
        db.insert_literature(
            project_id,
            title=f"Test Paper {i}",
            source="openalex" if i % 2 == 0 else "semantic_scholar",
            relevance_score=0.7 + (i % 10) * 0.03,
            included_in_review=(i < count // 2)
        )


def seed_experiments(db, project_id: int) -> None:
    """Seed experiment data for testing."""
    baselines = ["logistic_regression", "random_forest"]
    for name in baselines:
        exp_id = db.insert_experiment(
            project_id,
            experiment_name=name,
            experiment_type="baseline",
            random_seed=42
        )
        db.update_experiment_results(
            exp_id,
            metrics_json=json.dumps({"accuracy": 0.85}),
            runtime_seconds=30.0
        )

    # Add proposed method
    exp_id = db.insert_experiment(
        project_id,
        experiment_name="proposed_ensemble",
        experiment_type="proposed",
        random_seed=42
    )
    db.update_experiment_results(
        exp_id,
        metrics_json=json.dumps({"accuracy": 0.90}),
        runtime_seconds=60.0
    )
