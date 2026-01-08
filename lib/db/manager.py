"""Database manager for HOMER state persistence."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional


class DatabaseManager:
    """Manages SQLite database connections and operations for HOMER research state."""

    def __init__(self, db_path: Path | str = "research_state.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema if not exists."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with self.connection() as conn:
                conn.executescript(schema_path.read_text())

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection with row factory set
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Cursor after execution
        """
        with self.connection() as conn:
            return conn.execute(query, params)

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch a single row as dict.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Row as dict or None if no result
        """
        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as list of dicts.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of rows as dicts
        """
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def count(self, query: str, params: tuple = ()) -> int:
        """Execute a COUNT query and return the result.

        Args:
            query: SQL COUNT query
            params: Query parameters

        Returns:
            Count result
        """
        with self.connection() as conn:
            result = conn.execute(query, params).fetchone()
            return result[0] if result else 0

    def insert(self, table: str, data: dict) -> int:
        """Insert a row and return the ID.

        Args:
            table: Table name
            data: Column-value dict

        Returns:
            Inserted row ID
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self.connection() as conn:
            cursor = conn.execute(query, tuple(data.values()))
            return cursor.lastrowid

    def update(self, table: str, data: dict, where: str, params: tuple) -> int:
        """Update rows and return affected count.

        Args:
            table: Table name
            data: Column-value dict to update
            where: WHERE clause (without WHERE keyword)
            params: Parameters for WHERE clause

        Returns:
            Number of affected rows
        """
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        with self.connection() as conn:
            cursor = conn.execute(query, tuple(data.values()) + params)
            return cursor.rowcount

    def delete(self, table: str, where: str, params: tuple) -> int:
        """Delete rows and return affected count.

        Args:
            table: Table name
            where: WHERE clause (without WHERE keyword)
            params: Parameters for WHERE clause

        Returns:
            Number of deleted rows
        """
        query = f"DELETE FROM {table} WHERE {where}"
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount

    # =========================================================================
    # PROJECT OPERATIONS
    # =========================================================================

    def create_project(self, name: str, topic_json: str) -> int:
        """Create a new research project.

        Args:
            name: Unique project name
            topic_json: JSON string of ResearchTopic

        Returns:
            New project ID
        """
        return self.insert("research_projects", {
            "project_name": name,
            "topic_json": topic_json,
            "status": "initialized"
        })

    def get_project(self, project_id: int) -> Optional[dict]:
        """Get project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project dict or None
        """
        return self.fetch_one(
            "SELECT * FROM research_projects WHERE id = ?",
            (project_id,)
        )

    def get_project_by_name(self, name: str) -> Optional[dict]:
        """Get project by name.

        Args:
            name: Project name

        Returns:
            Project dict or None
        """
        return self.fetch_one(
            "SELECT * FROM research_projects WHERE project_name = ?",
            (name,)
        )

    def update_project_status(
        self,
        project_id: int,
        status: str,
        stage: Optional[str] = None,
        story_id: Optional[str] = None
    ) -> None:
        """Update project status and optionally stage/story.

        Args:
            project_id: Project ID
            status: New status
            stage: Current stage (optional)
            story_id: Current story ID (optional)
        """
        data = {"status": status}
        if stage is not None:
            data["current_stage"] = stage
        if story_id is not None:
            data["current_story_id"] = story_id
        self.update("research_projects", data, "id = ?", (project_id,))

    def increment_iteration(self, project_id: int) -> int:
        """Increment project iteration count.

        Args:
            project_id: Project ID

        Returns:
            New iteration count
        """
        with self.connection() as conn:
            conn.execute(
                "UPDATE research_projects SET iteration_count = iteration_count + 1 WHERE id = ?",
                (project_id,)
            )
            result = conn.execute(
                "SELECT iteration_count FROM research_projects WHERE id = ?",
                (project_id,)
            ).fetchone()
            return result[0] if result else 0

    def list_projects(self, status: Optional[str] = None) -> list[dict]:
        """List all projects, optionally filtered by status.

        Args:
            status: Filter by status (optional)

        Returns:
            List of project dicts
        """
        if status:
            return self.fetch_all(
                "SELECT * FROM research_projects WHERE status = ? ORDER BY updated_at DESC",
                (status,)
            )
        return self.fetch_all(
            "SELECT * FROM research_projects ORDER BY updated_at DESC"
        )

    # =========================================================================
    # LITERATURE OPERATIONS
    # =========================================================================

    def insert_literature(self, project_id: int, **kwargs) -> int:
        """Insert a literature entry.

        Args:
            project_id: Project ID
            **kwargs: Literature fields

        Returns:
            New literature ID
        """
        return self.insert("literature", {"project_id": project_id, **kwargs})

    def get_literature(self, literature_id: int) -> Optional[dict]:
        """Get literature entry by ID."""
        return self.fetch_one("SELECT * FROM literature WHERE id = ?", (literature_id,))

    def get_included_literature(self, project_id: int) -> list[dict]:
        """Get papers included in review.

        Args:
            project_id: Project ID

        Returns:
            List of included literature entries
        """
        return self.fetch_all(
            """SELECT * FROM literature
               WHERE project_id = ? AND included_in_review = 1
               ORDER BY relevance_score DESC""",
            (project_id,)
        )

    def get_literature_by_screening(
        self,
        project_id: int,
        screening_status: str
    ) -> list[dict]:
        """Get literature by screening status."""
        return self.fetch_all(
            "SELECT * FROM literature WHERE project_id = ? AND screening_status = ?",
            (project_id, screening_status)
        )

    def update_literature_screening(
        self,
        literature_id: int,
        status: str,
        included: bool = False,
        exclusion_reason: Optional[str] = None
    ) -> None:
        """Update literature screening status."""
        data = {"screening_status": status, "included_in_review": included}
        if exclusion_reason:
            data["exclusion_reason"] = exclusion_reason
        self.update("literature", data, "id = ?", (literature_id,))

    def count_literature(self, project_id: int, included_only: bool = False) -> int:
        """Count literature entries for project."""
        if included_only:
            return self.count(
                "SELECT COUNT(*) FROM literature WHERE project_id = ? AND included_in_review = 1",
                (project_id,)
            )
        return self.count(
            "SELECT COUNT(*) FROM literature WHERE project_id = ?",
            (project_id,)
        )

    # =========================================================================
    # HYPOTHESIS OPERATIONS
    # =========================================================================

    def insert_hypothesis(self, project_id: int, **kwargs) -> int:
        """Insert a hypothesis."""
        return self.insert("hypotheses", {"project_id": project_id, **kwargs})

    def get_hypotheses(self, project_id: int, tested_only: bool = False) -> list[dict]:
        """Get hypotheses for project."""
        if tested_only:
            return self.fetch_all(
                "SELECT * FROM hypotheses WHERE project_id = ? AND tested = 1 ORDER BY priority",
                (project_id,)
            )
        return self.fetch_all(
            "SELECT * FROM hypotheses WHERE project_id = ? ORDER BY priority",
            (project_id,)
        )

    def update_hypothesis_result(
        self,
        hypothesis_id: int,
        result: str,
        evidence_summary: str,
        supporting_experiments: Optional[str] = None
    ) -> None:
        """Update hypothesis test result."""
        self.update("hypotheses", {
            "tested": True,
            "result": result,
            "evidence_summary": evidence_summary,
            "supporting_experiments": supporting_experiments,
            "tested_at": datetime.now().isoformat()
        }, "id = ?", (hypothesis_id,))

    # =========================================================================
    # DATASET OPERATIONS
    # =========================================================================

    def insert_dataset(self, project_id: int, **kwargs) -> int:
        """Insert a dataset record."""
        return self.insert("datasets", {"project_id": project_id, **kwargs})

    def get_dataset(self, dataset_id: int) -> Optional[dict]:
        """Get dataset by ID."""
        return self.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))

    def get_datasets(self, project_id: int) -> list[dict]:
        """Get all datasets for project."""
        return self.fetch_all(
            "SELECT * FROM datasets WHERE project_id = ?",
            (project_id,)
        )

    def update_dataset_status(
        self,
        dataset_id: int,
        downloaded: Optional[bool] = None,
        eda_completed: Optional[bool] = None,
        preprocessing_completed: Optional[bool] = None
    ) -> None:
        """Update dataset processing status."""
        data = {}
        if downloaded is not None:
            data["downloaded"] = downloaded
        if eda_completed is not None:
            data["eda_completed"] = eda_completed
        if preprocessing_completed is not None:
            data["preprocessing_completed"] = preprocessing_completed
        if data:
            self.update("datasets", data, "id = ?", (dataset_id,))

    def update_dataset_schema(
        self,
        dataset_id: int,
        target_column: str,
        feature_columns: list[str],
        categorical_columns: list[str],
        numerical_columns: list[str]
    ) -> None:
        """Update dataset schema information."""
        self.update("datasets", {
            "target_column": target_column,
            "feature_columns": json.dumps(feature_columns),
            "categorical_columns": json.dumps(categorical_columns),
            "numerical_columns": json.dumps(numerical_columns)
        }, "id = ?", (dataset_id,))

    # =========================================================================
    # EXPERIMENT OPERATIONS
    # =========================================================================

    def insert_experiment(self, project_id: int, **kwargs) -> int:
        """Insert an experiment."""
        return self.insert("experiments", {"project_id": project_id, **kwargs})

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        """Get experiment by ID."""
        return self.fetch_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))

    def get_experiments(self, project_id: int) -> list[dict]:
        """Get all experiments for project."""
        return self.fetch_all(
            "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at",
            (project_id,)
        )

    def get_experiments_by_type(
        self,
        project_id: int,
        experiment_type: str
    ) -> list[dict]:
        """Get experiments by type (baseline, proposed, ablation, etc.)."""
        return self.fetch_all(
            "SELECT * FROM experiments WHERE project_id = ? AND experiment_type = ?",
            (project_id, experiment_type)
        )

    def get_completed_experiments(self, project_id: int) -> list[dict]:
        """Get completed experiments."""
        return self.fetch_all(
            "SELECT * FROM experiments WHERE project_id = ? AND status = 'completed'",
            (project_id,)
        )

    def update_experiment_status(
        self,
        experiment_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update experiment status."""
        data = {"status": status}
        if status == "running":
            data["started_at"] = datetime.now().isoformat()
        elif status in ("completed", "failed"):
            data["completed_at"] = datetime.now().isoformat()
        if error_message:
            data["error_message"] = error_message
        self.update("experiments", data, "id = ?", (experiment_id,))

    def update_experiment_results(
        self,
        experiment_id: int,
        metrics_json: str,
        runtime_seconds: float,
        memory_mb: Optional[float] = None,
        model_path: Optional[str] = None,
        predictions_path: Optional[str] = None
    ) -> None:
        """Update experiment with results."""
        data = {
            "metrics_json": metrics_json,
            "runtime_seconds": runtime_seconds,
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }
        if memory_mb is not None:
            data["memory_mb"] = memory_mb
        if model_path:
            data["model_path"] = model_path
        if predictions_path:
            data["predictions_path"] = predictions_path
        self.update("experiments", data, "id = ?", (experiment_id,))

    # =========================================================================
    # STATISTICAL TEST OPERATIONS
    # =========================================================================

    def insert_statistical_test(self, project_id: int, **kwargs) -> int:
        """Insert a statistical test result."""
        return self.insert("statistical_tests", {"project_id": project_id, **kwargs})

    def get_statistical_tests(self, project_id: int) -> list[dict]:
        """Get all statistical tests for project."""
        return self.fetch_all(
            "SELECT * FROM statistical_tests WHERE project_id = ?",
            (project_id,)
        )

    def get_significant_tests(self, project_id: int) -> list[dict]:
        """Get statistically significant tests."""
        return self.fetch_all(
            "SELECT * FROM statistical_tests WHERE project_id = ? AND significant = 1",
            (project_id,)
        )

    # =========================================================================
    # PAPER SECTION OPERATIONS
    # =========================================================================

    def insert_paper_section(self, project_id: int, **kwargs) -> int:
        """Insert a paper section."""
        return self.insert("paper_sections", {"project_id": project_id, **kwargs})

    def get_paper_section(
        self,
        project_id: int,
        section_name: str,
        version: Optional[int] = None
    ) -> Optional[dict]:
        """Get paper section by name, optionally specific version."""
        if version:
            return self.fetch_one(
                "SELECT * FROM paper_sections WHERE project_id = ? AND section_name = ? AND version = ?",
                (project_id, section_name, version)
            )
        return self.fetch_one(
            """SELECT * FROM paper_sections
               WHERE project_id = ? AND section_name = ?
               ORDER BY version DESC LIMIT 1""",
            (project_id, section_name)
        )

    def get_all_paper_sections(self, project_id: int) -> list[dict]:
        """Get all paper sections (latest versions)."""
        return self.fetch_all(
            """SELECT ps.* FROM paper_sections ps
               INNER JOIN (
                   SELECT section_name, MAX(version) as max_version
                   FROM paper_sections WHERE project_id = ?
                   GROUP BY section_name
               ) latest ON ps.section_name = latest.section_name
                        AND ps.version = latest.max_version
               WHERE ps.project_id = ?
               ORDER BY ps.section_order""",
            (project_id, project_id)
        )

    def update_paper_section(
        self,
        section_id: int,
        content: str,
        word_count: int,
        status: str
    ) -> None:
        """Update paper section content."""
        self.update("paper_sections", {
            "content": content,
            "word_count": word_count,
            "status": status
        }, "id = ?", (section_id,))

    # =========================================================================
    # FIGURE OPERATIONS
    # =========================================================================

    def insert_figure(self, project_id: int, **kwargs) -> int:
        """Insert a figure record."""
        return self.insert("figures", {"project_id": project_id, **kwargs})

    def get_figures(self, project_id: int, figure_type: Optional[str] = None) -> list[dict]:
        """Get figures, optionally filtered by type."""
        if figure_type:
            return self.fetch_all(
                "SELECT * FROM figures WHERE project_id = ? AND figure_type = ?",
                (project_id, figure_type)
            )
        return self.fetch_all(
            "SELECT * FROM figures WHERE project_id = ? ORDER BY figure_number",
            (project_id,)
        )

    def get_included_figures(self, project_id: int) -> list[dict]:
        """Get figures included in paper."""
        return self.fetch_all(
            "SELECT * FROM figures WHERE project_id = ? AND included_in_paper = 1 ORDER BY figure_number",
            (project_id,)
        )

    # =========================================================================
    # TABLE OPERATIONS
    # =========================================================================

    def insert_table(self, project_id: int, **kwargs) -> int:
        """Insert a table record."""
        return self.insert("tables", {"project_id": project_id, **kwargs})

    def get_tables(self, project_id: int) -> list[dict]:
        """Get all tables for project."""
        return self.fetch_all(
            "SELECT * FROM tables WHERE project_id = ? ORDER BY table_number",
            (project_id,)
        )

    def get_included_tables(self, project_id: int) -> list[dict]:
        """Get tables included in paper."""
        return self.fetch_all(
            "SELECT * FROM tables WHERE project_id = ? AND included_in_paper = 1 ORDER BY table_number",
            (project_id,)
        )

    # =========================================================================
    # AGENT LOG OPERATIONS
    # =========================================================================

    def log_agent_action(
        self,
        project_id: int,
        iteration: int,
        story_id: str,
        action: str,
        action_type: str,
        **kwargs
    ) -> int:
        """Log an agent action.

        Args:
            project_id: Project ID
            iteration: Current iteration number
            story_id: Current story ID
            action: Action description
            action_type: Type of action
            **kwargs: Additional log fields

        Returns:
            Log entry ID
        """
        return self.insert("agent_log", {
            "project_id": project_id,
            "iteration": iteration,
            "story_id": story_id,
            "action": action,
            "action_type": action_type,
            **kwargs
        })

    def get_agent_log(
        self,
        project_id: int,
        iteration: Optional[int] = None,
        limit: int = 100
    ) -> list[dict]:
        """Get agent log entries."""
        if iteration is not None:
            return self.fetch_all(
                "SELECT * FROM agent_log WHERE project_id = ? AND iteration = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, iteration, limit)
            )
        return self.fetch_all(
            "SELECT * FROM agent_log WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit)
        )

    def get_learnings(self, project_id: int) -> list[dict]:
        """Get all logged learnings for project."""
        return self.fetch_all(
            "SELECT * FROM agent_log WHERE project_id = ? AND learnings IS NOT NULL ORDER BY created_at",
            (project_id,)
        )

    # =========================================================================
    # CHECKPOINT OPERATIONS
    # =========================================================================

    def create_checkpoint(
        self,
        project_id: int,
        name: str,
        prd_json: str,
        progress_txt: str,
        git_hash: Optional[str] = None,
        checkpoint_type: str = "automatic",
        **kwargs
    ) -> int:
        """Create a checkpoint.

        Args:
            project_id: Project ID
            name: Checkpoint name
            prd_json: PRD state as JSON
            progress_txt: Progress.txt content
            git_hash: Git commit hash (optional)
            checkpoint_type: Type of checkpoint
            **kwargs: Additional checkpoint fields

        Returns:
            Checkpoint ID
        """
        return self.insert("checkpoints", {
            "project_id": project_id,
            "checkpoint_name": name,
            "checkpoint_type": checkpoint_type,
            "prd_json": prd_json,
            "progress_txt": progress_txt,
            "git_commit_hash": git_hash,
            **kwargs
        })

    def get_latest_checkpoint(self, project_id: int) -> Optional[dict]:
        """Get most recent checkpoint.

        Args:
            project_id: Project ID

        Returns:
            Checkpoint dict or None
        """
        return self.fetch_one(
            "SELECT * FROM checkpoints WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (project_id,)
        )

    def get_checkpoint_by_name(
        self,
        project_id: int,
        checkpoint_name: str
    ) -> Optional[dict]:
        """Get checkpoint by name."""
        return self.fetch_one(
            "SELECT * FROM checkpoints WHERE project_id = ? AND checkpoint_name = ?",
            (project_id, checkpoint_name)
        )

    def list_checkpoints(self, project_id: int) -> list[dict]:
        """List all checkpoints for project."""
        return self.fetch_all(
            "SELECT * FROM checkpoints WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,)
        )

    # =========================================================================
    # VIEW QUERIES
    # =========================================================================

    def get_project_summary(self, project_id: int) -> Optional[dict]:
        """Get project summary from view."""
        return self.fetch_one(
            "SELECT * FROM v_project_summary WHERE id = ?",
            (project_id,)
        )

    def get_experiment_results_view(self, project_id: int) -> list[dict]:
        """Get experiment results from view."""
        return self.fetch_all(
            "SELECT * FROM v_experiment_results WHERE project_id = ?",
            (project_id,)
        )


# Convenience function for getting a manager instance
def get_database_manager(db_path: Optional[Path | str] = None) -> DatabaseManager:
    """Get a DatabaseManager instance.

    Args:
        db_path: Optional path to database file

    Returns:
        DatabaseManager instance
    """
    if db_path:
        return DatabaseManager(db_path)
    return DatabaseManager()
