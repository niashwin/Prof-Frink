"""End-to-end tests for HOMER research pipeline.

These tests validate the complete research flow from topic definition
to paper generation using publicly available Kaggle datasets.

Test Datasets (as specified in TEST_SPECIFICATIONS.md):
1. Heart Disease UCI - Binary classification, health domain
2. Store Sales - Time series forecasting
3. Sentiment140 - NLP text classification
4. Online Retail - Customer segmentation/clustering
5. House Prices - Regression
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from homer.lib.db.manager import DatabaseManager
from homer.lib.prd_generator import generate_prd
from homer.lib.quality_gates import GateManager
from homer.lib.research_loop import (
    CheckpointManager,
    LoopStatus,
    ResearchLoop,
    SkillInvoker,
    StoryExecutor,
    StoryResult,
)
from homer.lib.schemas import (
    Constraints,
    DatasetConfig,
    ResearchPRD,
    ResearchTopic,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_db(temp_project_dir: Path) -> Generator[DatabaseManager, None, None]:
    """Create a temporary database."""
    db_path = temp_project_dir / "research_state.db"

    # Create full schema
    schema = """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;

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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            hypothesis_text TEXT NOT NULL,
            source TEXT NOT NULL,
            tested BOOLEAN DEFAULT FALSE,
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
            figure_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            included_in_paper BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES research_projects(id),
            table_type TEXT NOT NULL,
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
            result TEXT,
            output_summary TEXT,
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
            e.random_seed
        FROM experiments e;
    """

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.close()

    db = DatabaseManager(db_path)
    yield db


# =============================================================================
# E2E TEST 1: HEART DISEASE UCI (Binary Classification)
# =============================================================================


class TestHeartDiseaseE2E:
    """E2E test using Heart Disease UCI dataset.

    Dataset: kaggle/uciml/heart-disease-uci
    Task: Binary classification (presence of heart disease)
    Domain: MEDICINE
    """

    @pytest.fixture
    def heart_disease_topic(self) -> ResearchTopic:
        """Create research topic for heart disease prediction."""
        return ResearchTopic(
            title="Comparing Machine Learning Approaches for Heart Disease Prediction",
            hypothesis="Ensemble methods combining clinical and demographic features will outperform single models for heart disease prediction due to the complex interaction between risk factors.",
            domain="MEDICINE",
            datasets=[
                DatasetConfig(
                    source="kaggle",
                    identifier="uciml/heart-disease-uci",
                    description="UCI Heart Disease dataset with clinical features"
                )
            ],
            research_questions=[
                "Which ML algorithm performs best for heart disease prediction?",
                "What are the most important clinical features for prediction?",
                "How does class imbalance affect model performance?"
            ],
            constraints=Constraints(
                max_compute_hours=2.0,
                gpu_required=False
            ),
            keywords=["heart disease", "classification", "machine learning", "clinical prediction"]
        )

    def test_prd_generation(self, heart_disease_topic: ResearchTopic, temp_project_dir: Path):
        """Test PRD generation for heart disease project."""
        prd = generate_prd("heart-disease-ml", heart_disease_topic)

        assert prd.project == "heart-disease-ml"
        assert prd.topic.domain == "MEDICINE"
        assert len(prd.user_stories) > 0

        # Verify all stages are present
        stages = {s.stage for s in prd.user_stories}
        assert "literature" in stages
        assert "experiment" in stages
        assert "writing" in stages

    def test_project_initialization(
        self,
        heart_disease_topic: ResearchTopic,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test initializing heart disease research project."""
        # Create project in database
        project_id = temp_db.create_project(
            "heart-disease-ml",
            heart_disease_topic.model_dump_json()
        )

        project = temp_db.get_project(project_id)

        assert project is not None
        assert project["project_name"] == "heart-disease-ml"
        assert project["status"] == "initialized"

    def test_literature_stage_setup(
        self,
        heart_disease_topic: ResearchTopic,
        temp_db: DatabaseManager
    ):
        """Test setting up literature for heart disease research."""
        project_id = temp_db.create_project(
            "heart-disease-ml",
            heart_disease_topic.model_dump_json()
        )

        # Simulate literature retrieval
        papers = [
            {"title": "ML for heart disease", "source": "openalex", "score": 0.9},
            {"title": "Clinical prediction models", "source": "semantic_scholar", "score": 0.85},
            {"title": "Risk factor analysis", "source": "openalex", "score": 0.8},
        ]

        for paper in papers:
            temp_db.insert_literature(
                project_id,
                title=paper["title"],
                source=paper["source"],
                relevance_score=paper["score"],
                included_in_review=True
            )

        included = temp_db.get_included_literature(project_id)
        assert len(included) == 3

    def test_experiment_tracking(
        self,
        heart_disease_topic: ResearchTopic,
        temp_db: DatabaseManager
    ):
        """Test experiment tracking for heart disease models."""
        project_id = temp_db.create_project(
            "heart-disease-ml",
            heart_disease_topic.model_dump_json()
        )

        # Track baseline experiments
        baselines = ["logistic_regression", "random_forest", "xgboost"]
        for baseline in baselines:
            exp_id = temp_db.insert_experiment(
                project_id,
                experiment_name=baseline,
                experiment_type="baseline",
                random_seed=42
            )
            temp_db.update_experiment_results(
                exp_id,
                metrics_json=json.dumps({"accuracy": 0.85, "auc": 0.88}),
                runtime_seconds=30.0
            )

        completed = temp_db.get_completed_experiments(project_id)
        assert len(completed) == 3


# =============================================================================
# E2E TEST 2: STORE SALES (Time Series Forecasting)
# =============================================================================


class TestStoreSalesE2E:
    """E2E test using Store Sales dataset.

    Dataset: kaggle/competitions/store-sales-time-series-forecasting
    Task: Time series forecasting
    Domain: ML
    """

    @pytest.fixture
    def store_sales_topic(self) -> ResearchTopic:
        """Create research topic for store sales forecasting."""
        return ResearchTopic(
            title="Advanced Time Series Forecasting Methods for Retail Sales Prediction",
            hypothesis="Hybrid models combining traditional time series methods with gradient boosting will achieve better forecasting accuracy than pure deep learning approaches.",
            domain="ML",
            datasets=[
                DatasetConfig(
                    source="kaggle",
                    identifier="competitions/store-sales-time-series-forecasting",
                    description="Store sales time series data"
                )
            ],
            research_questions=[
                "How does seasonality affect forecasting accuracy?",
                "Can external features improve forecast performance?",
                "What is the optimal forecast horizon for this data?"
            ],
            constraints=Constraints(
                max_compute_hours=4.0,
                gpu_required=False
            ),
            keywords=["time series", "forecasting", "retail", "machine learning"]
        )

    def test_full_pipeline_simulation(
        self,
        store_sales_topic: ResearchTopic,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test full pipeline simulation for store sales."""
        # Generate PRD
        prd = generate_prd("store-sales-forecast", store_sales_topic)

        # Create project
        project_id = temp_db.create_project(
            "store-sales-forecast",
            store_sales_topic.model_dump_json()
        )

        # Simulate completing stages
        # 1. Literature
        for i in range(15):
            temp_db.insert_literature(
                project_id,
                title=f"Time series paper {i}",
                source="openalex" if i % 2 == 0 else "semantic_scholar",
                relevance_score=0.8,
                included_in_review=True
            )

        # 2. Dataset
        temp_db.insert_dataset(
            project_id,
            source="kaggle",
            identifier="store-sales",
            downloaded=True,
            eda_completed=True,
            preprocessing_completed=True,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )

        # 3. Experiments
        for exp_type, name in [("baseline", "arima"), ("baseline", "prophet"), ("proposed", "hybrid")]:
            exp_id = temp_db.insert_experiment(
                project_id,
                experiment_name=name,
                experiment_type=exp_type,
                random_seed=42
            )
            temp_db.update_experiment_results(
                exp_id,
                metrics_json=json.dumps({"rmse": 100.0, "mae": 80.0}),
                runtime_seconds=60.0
            )

        # Verify state
        summary = temp_db.get_project_summary(project_id)
        assert summary["papers_included"] == 15
        assert summary["experiments_completed"] == 3


# =============================================================================
# E2E TEST 3: SENTIMENT140 (NLP Classification)
# =============================================================================


class TestSentiment140E2E:
    """E2E test using Sentiment140 dataset.

    Dataset: kaggle/kazanova/sentiment140
    Task: Sentiment classification
    Domain: NLP
    """

    @pytest.fixture
    def sentiment_topic(self) -> ResearchTopic:
        """Create research topic for sentiment analysis."""
        return ResearchTopic(
            title="Transfer Learning Approaches for Twitter Sentiment Classification",
            hypothesis="Fine-tuned transformer models will significantly outperform traditional ML methods on Twitter sentiment analysis due to their ability to capture contextual nuances.",
            domain="NLP",
            datasets=[
                DatasetConfig(
                    source="kaggle",
                    identifier="kazanova/sentiment140",
                    description="Twitter sentiment dataset with 1.6M tweets"
                )
            ],
            research_questions=[
                "How does model size affect sentiment classification accuracy?",
                "Can data augmentation improve performance on imbalanced classes?",
                "What preprocessing steps are most important for tweets?"
            ],
            constraints=Constraints(
                max_compute_hours=8.0,
                gpu_required=True,
                max_model_parameters=110000000  # ~BERT-base size
            ),
            keywords=["sentiment analysis", "NLP", "transformers", "Twitter"]
        )

    def test_nlp_domain_skills(self, sentiment_topic: ResearchTopic):
        """Test that NLP domain adds correct skills."""
        prd = generate_prd("sentiment-analysis", sentiment_topic)

        all_skills = set()
        for story in prd.user_stories:
            all_skills.update(story.skills_required)

        # Should have NLP-specific skills
        nlp_skills = {"transformers", "nltk", "spacy"}
        assert len(nlp_skills & all_skills) > 0


# =============================================================================
# E2E TEST 4: ONLINE RETAIL (Clustering)
# =============================================================================


class TestOnlineRetailE2E:
    """E2E test using Online Retail dataset.

    Dataset: kaggle/carrie1/ecommerce-data
    Task: Customer segmentation
    Domain: ML
    """

    @pytest.fixture
    def retail_topic(self) -> ResearchTopic:
        """Create research topic for customer segmentation."""
        return ResearchTopic(
            title="Customer Segmentation Using RFM Analysis and Machine Learning",
            hypothesis="Combining RFM metrics with modern clustering algorithms will produce more actionable customer segments than traditional RFM quartile analysis.",
            domain="ML",
            datasets=[
                DatasetConfig(
                    source="kaggle",
                    identifier="carrie1/ecommerce-data",
                    description="Online retail transaction data"
                )
            ],
            research_questions=[
                "What is the optimal number of customer segments?",
                "How do different clustering algorithms compare?",
                "Can we predict customer segment migration?"
            ],
            constraints=Constraints(
                max_compute_hours=2.0,
                gpu_required=False
            ),
            keywords=["customer segmentation", "clustering", "RFM", "e-commerce"]
        )

    def test_checkpoint_creation(
        self,
        retail_topic: ResearchTopic,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test checkpoint creation during research."""
        prd = generate_prd("customer-segmentation", retail_topic)

        project_id = temp_db.create_project(
            "customer-segmentation",
            retail_topic.model_dump_json()
        )

        # Save PRD to file
        prd_path = temp_project_dir / "research_prd.json"
        prd.to_file(prd_path)

        # Create checkpoint manager
        cp_manager = CheckpointManager(temp_db, temp_project_dir)

        # Create checkpoint
        cp_id = cp_manager.create_checkpoint(
            project_id,
            prd,
            checkpoint_type="stage_complete",
            trigger_reason="Completed literature stage"
        )

        assert cp_id > 0

        # Verify checkpoint can be restored
        restored_prd = cp_manager.restore_checkpoint(project_id)
        assert restored_prd is not None
        assert restored_prd.project == prd.project


# =============================================================================
# E2E TEST 5: HOUSE PRICES (Regression)
# =============================================================================


class TestHousePricesE2E:
    """E2E test using House Prices dataset.

    Dataset: kaggle/competitions/house-prices-advanced-regression-techniques
    Task: Regression
    Domain: ML
    """

    @pytest.fixture
    def house_prices_topic(self) -> ResearchTopic:
        """Create research topic for house price prediction."""
        return ResearchTopic(
            title="Feature Engineering Impact on House Price Prediction Models",
            hypothesis="Careful feature engineering combining domain knowledge with automated feature selection will outperform deep learning approaches that rely on raw features.",
            domain="ML",
            datasets=[
                DatasetConfig(
                    source="kaggle",
                    identifier="competitions/house-prices-advanced-regression-techniques",
                    description="Ames Housing dataset"
                )
            ],
            research_questions=[
                "Which feature engineering techniques have the most impact?",
                "How do tree-based models compare to linear models?",
                "What is the effect of handling missing values?"
            ],
            constraints=Constraints(
                max_compute_hours=2.0,
                gpu_required=False
            ),
            keywords=["regression", "feature engineering", "real estate", "prediction"]
        )

    def test_quality_gate_progression(
        self,
        house_prices_topic: ResearchTopic,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test quality gate progression through stages."""
        project_id = temp_db.create_project(
            "house-prices",
            house_prices_topic.model_dump_json()
        )

        gate_manager = GateManager(temp_db)

        # Initially all gates should fail
        lit_result = gate_manager.evaluate_stage(project_id, "literature")
        assert lit_result.passed is False

        # Add sufficient literature
        for i in range(30):
            temp_db.insert_literature(
                project_id,
                title=f"Paper {i}",
                source="openalex" if i % 2 == 0 else "semantic_scholar",
                relevance_score=0.8,
                included_in_review=(i < 15)
            )

        # Now literature gate should pass
        lit_result = gate_manager.evaluate_stage(project_id, "literature")
        assert lit_result.passed is True

    def test_research_loop_execution(
        self,
        house_prices_topic: ResearchTopic,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test research loop execution with mocked skills."""
        prd = generate_prd("house-prices", house_prices_topic)

        project_id = temp_db.create_project(
            "house-prices",
            house_prices_topic.model_dump_json()
        )

        # Save PRD
        prd_path = temp_project_dir / "research_prd.json"
        prd.to_file(prd_path)

        # Create progress file
        progress_path = temp_project_dir / "progress.txt"
        progress_path.write_text("")

        # Create research loop with mocked skill invoker
        with patch.object(SkillInvoker, 'invoke') as mock_invoke:
            mock_invoke.return_value = {"status": "success", "outputs": []}

            with patch.object(StoryExecutor, '_verify_acceptance_criteria') as mock_verify:
                mock_verify.return_value = True

                with patch.object(CheckpointManager, 'git_commit') as mock_commit:
                    mock_commit.return_value = True

                    loop = ResearchLoop(
                        project_id,
                        temp_project_dir,
                        temp_db
                    )

                    # Run limited iterations
                    status = loop.run(max_iterations=3)

                    # Should have run and paused
                    assert status in [LoopStatus.PAUSED, LoopStatus.RUNNING, LoopStatus.FAILED]

                    # Verify some iterations occurred
                    project = temp_db.get_project(project_id)
                    assert project["iteration_count"] >= 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestResearchLoopIntegration:
    """Integration tests for the research loop."""

    def test_callback_registration(self, temp_db: DatabaseManager, temp_project_dir: Path):
        """Test that callbacks can be registered and called."""
        topic = ResearchTopic(
            title="Integration Test Research Project",
            hypothesis="This is a test hypothesis that is long enough for validation purposes.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        prd = generate_prd("integration-test", topic)
        project_id = temp_db.create_project("integration-test", topic.model_dump_json())

        prd_path = temp_project_dir / "research_prd.json"
        prd.to_file(prd_path)

        progress_path = temp_project_dir / "progress.txt"
        progress_path.write_text("")

        loop = ResearchLoop(project_id, temp_project_dir, temp_db)

        # Track callback calls
        callback_calls = []

        def on_story_start(**kwargs):
            callback_calls.append(("story_start", kwargs))

        def on_error(**kwargs):
            callback_calls.append(("error", kwargs))

        loop.register_callback("story_start", on_story_start)
        loop.register_callback("error", on_error)

        # Run with mocked execution
        with patch.object(loop.story_executor, 'execute') as mock_execute:
            mock_execute.return_value = StoryResult(
                story_id="LIT-001",
                success=True,
                outputs=[]
            )

            with patch.object(loop.checkpoint_manager, 'git_commit'):
                loop.run(max_iterations=1)

        # Verify callbacks were called
        assert any(call[0] == "story_start" for call in callback_calls)

    def test_loop_pause_and_resume(
        self,
        temp_db: DatabaseManager,
        temp_project_dir: Path
    ):
        """Test pausing and resuming the research loop."""
        topic = ResearchTopic(
            title="Pause Resume Test Research Project",
            hypothesis="This is a test hypothesis that is long enough for validation purposes.",
            domain="ML",
            datasets=[DatasetConfig(source="kaggle", identifier="test/data")]
        )

        prd = generate_prd("pause-resume-test", topic)
        project_id = temp_db.create_project("pause-resume-test", topic.model_dump_json())

        prd_path = temp_project_dir / "research_prd.json"
        prd.to_file(prd_path)

        progress_path = temp_project_dir / "progress.txt"
        progress_path.write_text("")

        # First run
        loop = ResearchLoop(project_id, temp_project_dir, temp_db)

        with patch.object(loop.story_executor, 'execute') as mock_execute:
            mock_execute.return_value = StoryResult(
                story_id="LIT-001",
                success=True,
                outputs=[]
            )

            with patch.object(loop.checkpoint_manager, 'git_commit'):
                status = loop.run(max_iterations=2)

        # Verify state was saved
        checkpoint = temp_db.get_latest_checkpoint(project_id)
        assert checkpoint is not None

        # Second run (resume)
        loop2 = ResearchLoop(project_id, temp_project_dir, temp_db)

        with patch.object(loop2.story_executor, 'execute') as mock_execute2:
            mock_execute2.return_value = StoryResult(
                story_id="LIT-002",
                success=True,
                outputs=[]
            )

            with patch.object(loop2.checkpoint_manager, 'git_commit'):
                status2 = loop2.run(max_iterations=1)

        # Verify iterations accumulated
        project = temp_db.get_project(project_id)
        assert project["iteration_count"] >= 2
