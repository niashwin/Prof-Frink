-- FRINK Database Schema
-- Hypothesis-Oriented Machine Experimentation & Research
--
-- This schema defines the state persistence layer for autonomous research projects.
-- All tables support the Ralph loop pattern with checkpointing and progress tracking.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ============================================================================
-- CORE PROJECT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS research_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL UNIQUE,
    topic_json TEXT NOT NULL,
    status TEXT CHECK(status IN ('initialized', 'in_progress', 'completed', 'failed', 'paused')) DEFAULT 'initialized',
    current_stage TEXT CHECK(current_stage IN ('literature', 'hypothesis', 'data', 'experiment', 'analysis', 'visualization', 'writing', 'review', NULL)),
    current_story_id TEXT,
    iteration_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_projects_status ON research_projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_stage ON research_projects(current_stage);

-- ============================================================================
-- LITERATURE REVIEW TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS literature (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Identifiers
    doi TEXT,
    pmid TEXT,
    arxiv_id TEXT,
    openalex_id TEXT,

    -- Bibliographic data
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    journal TEXT,
    volume TEXT,
    pages TEXT,

    -- Source and content
    source TEXT CHECK(source IN ('openalex', 'pubmed', 'biorxiv', 'arxiv', 'semantic_scholar', 'manual')) NOT NULL,
    abstract TEXT,
    keywords TEXT,
    full_text_available BOOLEAN DEFAULT FALSE,
    pdf_path TEXT,

    -- Relevance and screening
    relevance_score REAL CHECK(relevance_score >= 0 AND relevance_score <= 1),
    cited_count INTEGER DEFAULT 0,
    screening_status TEXT CHECK(screening_status IN ('pending', 'title_screen', 'abstract_screen', 'full_text', 'included', 'excluded')) DEFAULT 'pending',
    included_in_review BOOLEAN DEFAULT FALSE,
    exclusion_reason TEXT,

    -- Notes and metadata
    notes TEXT,
    bibtex TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, doi),
    UNIQUE(project_id, pmid),
    UNIQUE(project_id, openalex_id)
);

CREATE INDEX IF NOT EXISTS idx_literature_project ON literature(project_id);
CREATE INDEX IF NOT EXISTS idx_literature_relevance ON literature(project_id, relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_literature_included ON literature(project_id, included_in_review);
CREATE INDEX IF NOT EXISTS idx_literature_screening ON literature(project_id, screening_status);

-- ============================================================================
-- HYPOTHESIS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Hypothesis content
    hypothesis_text TEXT NOT NULL,
    operationalization TEXT,

    -- Classification
    source TEXT CHECK(source IN ('initial', 'literature', 'hypogenic', 'refined', 'ablation')) NOT NULL,
    hypothesis_type TEXT CHECK(hypothesis_type IN ('primary', 'secondary', 'exploratory')) DEFAULT 'primary',

    -- Testing status
    testable BOOLEAN DEFAULT TRUE,
    tested BOOLEAN DEFAULT FALSE,
    result TEXT CHECK(result IN ('supported', 'partially_supported', 'refuted', 'inconclusive', NULL)),

    -- Evidence
    evidence_summary TEXT,
    supporting_experiments TEXT,

    -- Metadata
    priority INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    tested_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_project ON hypotheses(project_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_tested ON hypotheses(project_id, tested);

-- ============================================================================
-- DATASET TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Source identification
    source TEXT CHECK(source IN ('kaggle', 'uci', 'huggingface', 'openml', 'zenodo', 'custom')) NOT NULL,
    identifier TEXT NOT NULL,
    name TEXT,
    description TEXT,
    url TEXT,

    -- Local storage
    local_path TEXT,
    raw_path TEXT,
    processed_path TEXT,

    -- Dataset characteristics
    rows_count INTEGER,
    columns_count INTEGER,
    size_bytes INTEGER,
    file_format TEXT,

    -- Schema information
    target_column TEXT,
    feature_columns TEXT,  -- JSON array
    categorical_columns TEXT,  -- JSON array
    numerical_columns TEXT,  -- JSON array

    -- Processing status
    downloaded BOOLEAN DEFAULT FALSE,
    eda_completed BOOLEAN DEFAULT FALSE,
    preprocessing_completed BOOLEAN DEFAULT FALSE,

    -- Split information
    split_seed INTEGER,
    train_size REAL,
    val_size REAL,
    test_size REAL,

    -- Metadata
    license TEXT,
    citation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, source, identifier)
);

CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id);

-- ============================================================================
-- EXPERIMENT TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL,

    -- Experiment identification
    experiment_name TEXT NOT NULL,
    experiment_type TEXT CHECK(experiment_type IN ('baseline', 'proposed', 'ablation', 'hyperparameter', 'sensitivity')) NOT NULL,

    -- Model configuration
    model_type TEXT,
    model_class TEXT,
    model_config_json TEXT,
    hyperparameters_json TEXT,

    -- Reproducibility
    random_seed INTEGER,
    framework TEXT,
    framework_version TEXT,

    -- Results
    metrics_json TEXT,
    predictions_path TEXT,
    model_path TEXT,

    -- Resource usage
    runtime_seconds REAL,
    memory_mb REAL,
    gpu_used BOOLEAN DEFAULT FALSE,

    -- Status
    status TEXT CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
    error_message TEXT,

    -- Artifacts
    notebook_path TEXT,
    log_path TEXT,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(project_id, experiment_type);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(project_id, status);

-- ============================================================================
-- STATISTICAL ANALYSIS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS statistical_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Comparison setup
    experiment_id_1 INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    experiment_id_2 INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    comparison_name TEXT,

    -- Test specification
    test_name TEXT NOT NULL,
    test_type TEXT CHECK(test_type IN ('parametric', 'non_parametric', 'bayesian', 'bootstrap')),
    metric_compared TEXT,

    -- Hypotheses
    null_hypothesis TEXT,
    alternative_hypothesis TEXT,

    -- Results
    test_statistic REAL,
    degrees_freedom REAL,
    p_value REAL,

    -- Effect size
    effect_size REAL,
    effect_size_type TEXT CHECK(effect_size_type IN ('cohens_d', 'hedges_g', 'glass_delta', 'eta_squared', 'omega_squared', 'r', 'odds_ratio', 'risk_ratio', 'other')),
    effect_size_interpretation TEXT CHECK(effect_size_interpretation IN ('negligible', 'small', 'medium', 'large', 'very_large')),

    -- Confidence interval
    confidence_level REAL DEFAULT 0.95,
    ci_lower REAL,
    ci_upper REAL,

    -- Sample information
    sample_size_1 INTEGER,
    sample_size_2 INTEGER,

    -- Assumptions
    assumptions_checked BOOLEAN DEFAULT FALSE,
    assumptions_met BOOLEAN,
    assumptions_notes TEXT,

    -- Conclusion
    significant BOOLEAN,
    conclusion TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stats_project ON statistical_tests(project_id);
CREATE INDEX IF NOT EXISTS idx_stats_significant ON statistical_tests(project_id, significant);

-- ============================================================================
-- PAPER WRITING TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Section identification
    section_name TEXT CHECK(section_name IN (
        'abstract', 'introduction', 'related_work', 'methodology',
        'experiments', 'results', 'discussion', 'conclusion',
        'appendix', 'acknowledgments', 'references'
    )) NOT NULL,
    section_order INTEGER,

    -- Content
    version INTEGER DEFAULT 1,
    content TEXT,
    outline TEXT,

    -- Metrics
    word_count INTEGER,
    citation_count INTEGER,
    figure_count INTEGER,
    table_count INTEGER,

    -- Status
    status TEXT CHECK(status IN ('outline', 'draft', 'revised', 'final')) DEFAULT 'outline',

    -- Review
    review_notes TEXT,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, section_name, version)
);

CREATE INDEX IF NOT EXISTS idx_sections_project ON paper_sections(project_id);
CREATE INDEX IF NOT EXISTS idx_sections_status ON paper_sections(project_id, status);

-- ============================================================================
-- FIGURES AND TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS figures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Identification
    figure_number INTEGER,
    figure_type TEXT CHECK(figure_type IN (
        'results', 'comparison', 'ablation', 'architecture', 'method',
        'data_viz', 'eda', 'prisma', 'confusion_matrix', 'roc', 'learning_curve',
        'feature_importance', 'shap', 'custom'
    )) NOT NULL,

    -- Content
    caption TEXT,
    alt_text TEXT,

    -- File information
    file_path TEXT NOT NULL,
    file_format TEXT CHECK(file_format IN ('png', 'pdf', 'svg', 'eps')) DEFAULT 'png',

    -- Dimensions
    width_inches REAL,
    height_inches REAL,
    dpi INTEGER DEFAULT 300,

    -- Usage
    included_in_paper BOOLEAN DEFAULT FALSE,
    section_name TEXT,

    -- Metadata
    source_data TEXT,
    generation_code TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_figures_project ON figures(project_id);
CREATE INDEX IF NOT EXISTS idx_figures_type ON figures(project_id, figure_type);

CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Identification
    table_number INTEGER,
    table_type TEXT CHECK(table_type IN (
        'results', 'comparison', 'ablation', 'dataset_stats', 'hyperparameters',
        'statistical_tests', 'literature_summary', 'custom'
    )) NOT NULL,

    -- Content
    caption TEXT,
    content_csv TEXT,
    content_latex TEXT,

    -- Usage
    included_in_paper BOOLEAN DEFAULT FALSE,
    section_name TEXT,

    -- Metadata
    row_count INTEGER,
    col_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tables_project ON tables(project_id);

-- ============================================================================
-- AGENT ACTIVITY LOGGING
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Loop tracking
    iteration INTEGER NOT NULL,
    story_id TEXT,

    -- Action details
    action TEXT NOT NULL,
    action_type TEXT CHECK(action_type IN ('story_start', 'skill_invoke', 'quality_gate', 'commit', 'checkpoint', 'error', 'complete')),

    -- Skill information
    skill_used TEXT,
    skill_params_json TEXT,

    -- Results
    result TEXT CHECK(result IN ('success', 'partial', 'failed', 'skipped')),
    output_summary TEXT,
    error_message TEXT,

    -- Learnings
    learnings TEXT,
    patterns_discovered TEXT,

    -- Timing
    duration_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_project ON agent_log(project_id);
CREATE INDEX IF NOT EXISTS idx_log_iteration ON agent_log(project_id, iteration);
CREATE INDEX IF NOT EXISTS idx_log_story ON agent_log(project_id, story_id);

-- ============================================================================
-- CHECKPOINTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,

    -- Checkpoint identification
    checkpoint_name TEXT NOT NULL,
    checkpoint_type TEXT CHECK(checkpoint_type IN ('manual', 'automatic', 'stage_complete', 'error_recovery')) DEFAULT 'automatic',

    -- State snapshots
    prd_json TEXT NOT NULL,
    progress_txt TEXT,

    -- Git state
    git_branch TEXT,
    git_commit_hash TEXT,

    -- Progress metrics
    stage_completed TEXT,
    stories_completed INTEGER,
    stories_total INTEGER,

    -- Metadata
    trigger_reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, checkpoint_name)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_project ON checkpoints(project_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_time ON checkpoints(project_id, created_at DESC);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- ============================================================================

CREATE TRIGGER IF NOT EXISTS update_project_timestamp
AFTER UPDATE ON research_projects
FOR EACH ROW
BEGIN
    UPDATE research_projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_section_timestamp
AFTER UPDATE ON paper_sections
FOR EACH ROW
BEGIN
    UPDATE paper_sections SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

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
    e.model_type,
    e.random_seed,
    e.status,
    e.metrics_json,
    e.runtime_seconds,
    d.name as dataset_name,
    h.hypothesis_text
FROM experiments e
LEFT JOIN datasets d ON e.dataset_id = d.id
LEFT JOIN hypotheses h ON e.hypothesis_id = h.id;
