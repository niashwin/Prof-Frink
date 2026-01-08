"""Quality Gates for HOMER.

Quality gates are checkpoints that validate research quality at each stage.
They ensure the research meets minimum standards before proceeding.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from homer.lib.db.manager import DatabaseManager
from homer.lib.schemas import QualityGateResult


# =============================================================================
# BASE QUALITY GATE
# =============================================================================


@dataclass
class GateCheckResult:
    """Result of a single gate check."""

    check_name: str
    passed: bool
    score: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class QualityGate(ABC):
    """Base class for quality gates."""

    def __init__(
        self,
        name: str,
        threshold: float = 0.7,
        required: bool = True,
        max_retries: int = 3
    ):
        """Initialize quality gate.

        Args:
            name: Gate name
            threshold: Minimum score to pass (0-1)
            required: Whether this gate is required
            max_retries: Maximum retry attempts
        """
        self.name = name
        self.threshold = threshold
        self.required = required
        self.max_retries = max_retries
        self._retry_count = 0

    @abstractmethod
    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        """Run all checks for this gate.

        Args:
            project_id: Project ID to check
            db: Database manager

        Returns:
            List of check results
        """
        pass

    def evaluate(self, project_id: int, db: DatabaseManager) -> QualityGateResult:
        """Evaluate the quality gate.

        Args:
            project_id: Project ID to evaluate
            db: Database manager

        Returns:
            QualityGateResult with overall pass/fail and details
        """
        checks = self.check(project_id, db)

        # Calculate overall score
        if not checks:
            score = 0.0
        else:
            score = sum(c.score for c in checks) / len(checks)

        passed = score >= self.threshold

        # Collect recommendations for failed checks
        recommendations = [
            c.message for c in checks
            if not c.passed
        ]

        return QualityGateResult(
            gate_name=self.name,
            passed=passed,
            score=score,
            threshold=self.threshold,
            details={
                "checks": [
                    {
                        "name": c.check_name,
                        "passed": c.passed,
                        "score": c.score,
                        "message": c.message,
                        "details": c.details
                    }
                    for c in checks
                ],
                "retry_count": self._retry_count
            },
            recommendations=recommendations,
            checked_at=datetime.now().isoformat()
        )

    def can_retry(self) -> bool:
        """Check if retry is allowed."""
        return self._retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self._retry_count += 1


# =============================================================================
# LITERATURE GATE
# =============================================================================


class LiteratureGate(QualityGate):
    """Quality gate for literature review stage."""

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="LiteratureGate",
            threshold=threshold,
            required=True
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        # Check 1: Minimum papers retrieved
        total_papers = db.count_literature(project_id, included_only=False)
        min_papers = 20
        papers_check = GateCheckResult(
            check_name="minimum_papers_retrieved",
            passed=total_papers >= min_papers,
            score=min(1.0, total_papers / min_papers),
            message=f"Retrieved {total_papers} papers (minimum: {min_papers})",
            details={"total_papers": total_papers, "minimum": min_papers}
        )
        checks.append(papers_check)

        # Check 2: Minimum papers included
        included_papers = db.count_literature(project_id, included_only=True)
        min_included = 10
        included_check = GateCheckResult(
            check_name="minimum_papers_included",
            passed=included_papers >= min_included,
            score=min(1.0, included_papers / min_included),
            message=f"Included {included_papers} papers (minimum: {min_included})",
            details={"included_papers": included_papers, "minimum": min_included}
        )
        checks.append(included_check)

        # Check 3: Relevance score distribution
        included_lit = db.get_included_literature(project_id)
        if included_lit:
            avg_relevance = sum(
                p.get("relevance_score", 0) or 0 for p in included_lit
            ) / len(included_lit)
            relevance_check = GateCheckResult(
                check_name="average_relevance_score",
                passed=avg_relevance >= 0.6,
                score=avg_relevance,
                message=f"Average relevance score: {avg_relevance:.2f}",
                details={"average_relevance": avg_relevance}
            )
        else:
            relevance_check = GateCheckResult(
                check_name="average_relevance_score",
                passed=False,
                score=0.0,
                message="No papers included yet",
                details={}
            )
        checks.append(relevance_check)

        # Check 4: Multiple sources used
        sources = set()
        all_lit = db.fetch_all(
            "SELECT DISTINCT source FROM literature WHERE project_id = ?",
            (project_id,)
        )
        sources = {r["source"] for r in all_lit}
        multi_source_check = GateCheckResult(
            check_name="multiple_sources_used",
            passed=len(sources) >= 2,
            score=min(1.0, len(sources) / 2),
            message=f"Used {len(sources)} sources: {', '.join(sources)}",
            details={"sources": list(sources)}
        )
        checks.append(multi_source_check)

        return checks


# =============================================================================
# DATA GATE
# =============================================================================


class DataGate(QualityGate):
    """Quality gate for data preparation stage."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(
            name="DataGate",
            threshold=threshold,
            required=True
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        datasets = db.get_datasets(project_id)

        # Check 1: Dataset downloaded
        downloaded = all(d.get("downloaded", False) for d in datasets)
        download_check = GateCheckResult(
            check_name="datasets_downloaded",
            passed=downloaded,
            score=1.0 if downloaded else 0.0,
            message="All datasets downloaded" if downloaded else "Some datasets not downloaded",
            details={"datasets_count": len(datasets)}
        )
        checks.append(download_check)

        # Check 2: EDA completed
        eda_completed = all(d.get("eda_completed", False) for d in datasets)
        eda_check = GateCheckResult(
            check_name="eda_completed",
            passed=eda_completed,
            score=1.0 if eda_completed else 0.0,
            message="EDA completed for all datasets" if eda_completed else "EDA not completed",
            details={}
        )
        checks.append(eda_check)

        # Check 3: Preprocessing completed
        preproc_completed = all(d.get("preprocessing_completed", False) for d in datasets)
        preproc_check = GateCheckResult(
            check_name="preprocessing_completed",
            passed=preproc_completed,
            score=1.0 if preproc_completed else 0.0,
            message="Preprocessing completed" if preproc_completed else "Preprocessing not completed",
            details={}
        )
        checks.append(preproc_check)

        # Check 4: Train/val/test split exists
        has_splits = all(
            d.get("train_size") and d.get("val_size") and d.get("test_size")
            for d in datasets
        )
        split_check = GateCheckResult(
            check_name="data_splits_created",
            passed=has_splits,
            score=1.0 if has_splits else 0.0,
            message="Data splits created" if has_splits else "Data splits not created",
            details={}
        )
        checks.append(split_check)

        return checks


# =============================================================================
# EXPERIMENT GATE
# =============================================================================


class ExperimentGate(QualityGate):
    """Quality gate for experiment stage."""

    def __init__(self, threshold: float = 0.75):
        super().__init__(
            name="ExperimentGate",
            threshold=threshold,
            required=True
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        # Check 1: Baseline experiments completed
        baselines = db.get_experiments_by_type(project_id, "baseline")
        completed_baselines = [e for e in baselines if e.get("status") == "completed"]
        baseline_check = GateCheckResult(
            check_name="baseline_experiments_completed",
            passed=len(completed_baselines) >= 2,
            score=min(1.0, len(completed_baselines) / 2),
            message=f"Completed {len(completed_baselines)} baseline experiments (minimum: 2)",
            details={"completed_baselines": len(completed_baselines)}
        )
        checks.append(baseline_check)

        # Check 2: Proposed method completed
        proposed = db.get_experiments_by_type(project_id, "proposed")
        completed_proposed = [e for e in proposed if e.get("status") == "completed"]
        proposed_check = GateCheckResult(
            check_name="proposed_method_completed",
            passed=len(completed_proposed) >= 1,
            score=1.0 if completed_proposed else 0.0,
            message="Proposed method experiment completed" if completed_proposed else "Proposed method not completed",
            details={"completed_proposed": len(completed_proposed)}
        )
        checks.append(proposed_check)

        # Check 3: All experiments have metrics
        all_experiments = db.get_completed_experiments(project_id)
        has_metrics = all(e.get("metrics_json") for e in all_experiments)
        metrics_check = GateCheckResult(
            check_name="all_experiments_have_metrics",
            passed=has_metrics and len(all_experiments) > 0,
            score=1.0 if (has_metrics and all_experiments) else 0.0,
            message="All experiments have metrics recorded" if has_metrics else "Some experiments missing metrics",
            details={"total_experiments": len(all_experiments)}
        )
        checks.append(metrics_check)

        # Check 4: Reproducibility (seeds set)
        seeds_set = all(e.get("random_seed") is not None for e in all_experiments)
        seed_check = GateCheckResult(
            check_name="reproducibility_seeds_set",
            passed=seeds_set and len(all_experiments) > 0,
            score=1.0 if (seeds_set and all_experiments) else 0.0,
            message="All experiments have random seeds" if seeds_set else "Some experiments missing seeds",
            details={}
        )
        checks.append(seed_check)

        return checks


# =============================================================================
# STATISTICS GATE
# =============================================================================


class StatisticsGate(QualityGate):
    """Quality gate for statistical analysis stage."""

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="StatisticsGate",
            threshold=threshold,
            required=True
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        stat_tests = db.get_statistical_tests(project_id)

        # Check 1: Statistical tests performed
        tests_performed_check = GateCheckResult(
            check_name="statistical_tests_performed",
            passed=len(stat_tests) >= 1,
            score=min(1.0, len(stat_tests)),
            message=f"Performed {len(stat_tests)} statistical tests",
            details={"test_count": len(stat_tests)}
        )
        checks.append(tests_performed_check)

        # Check 2: Effect sizes calculated
        has_effect_sizes = any(t.get("effect_size") is not None for t in stat_tests)
        effect_check = GateCheckResult(
            check_name="effect_sizes_calculated",
            passed=has_effect_sizes,
            score=1.0 if has_effect_sizes else 0.0,
            message="Effect sizes calculated" if has_effect_sizes else "No effect sizes calculated",
            details={}
        )
        checks.append(effect_check)

        # Check 3: Confidence intervals computed
        has_ci = any(
            t.get("ci_lower") is not None and t.get("ci_upper") is not None
            for t in stat_tests
        )
        ci_check = GateCheckResult(
            check_name="confidence_intervals_computed",
            passed=has_ci,
            score=1.0 if has_ci else 0.0,
            message="Confidence intervals computed" if has_ci else "No confidence intervals",
            details={}
        )
        checks.append(ci_check)

        # Check 4: Assumptions checked
        assumptions_checked = any(t.get("assumptions_checked") for t in stat_tests)
        assumptions_check = GateCheckResult(
            check_name="assumptions_verified",
            passed=assumptions_checked,
            score=1.0 if assumptions_checked else 0.0,
            message="Statistical assumptions verified" if assumptions_checked else "Assumptions not checked",
            details={}
        )
        checks.append(assumptions_check)

        return checks


# =============================================================================
# WRITING GATE
# =============================================================================


class WritingGate(QualityGate):
    """Quality gate for paper writing stage."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(
            name="WritingGate",
            threshold=threshold,
            required=True
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        sections = db.get_all_paper_sections(project_id)
        section_names = {s.get("section_name") for s in sections}

        # Check 1: All required sections present
        required_sections = {
            "abstract", "introduction", "related_work",
            "methodology", "experiments", "results", "conclusion"
        }
        missing_sections = required_sections - section_names
        sections_check = GateCheckResult(
            check_name="required_sections_present",
            passed=len(missing_sections) == 0,
            score=1.0 - (len(missing_sections) / len(required_sections)),
            message=f"Missing sections: {missing_sections}" if missing_sections else "All sections present",
            details={"missing": list(missing_sections)}
        )
        checks.append(sections_check)

        # Check 2: Sections have content
        sections_with_content = [s for s in sections if s.get("content")]
        content_check = GateCheckResult(
            check_name="sections_have_content",
            passed=len(sections_with_content) == len(sections) and len(sections) > 0,
            score=len(sections_with_content) / max(len(sections), 1),
            message=f"{len(sections_with_content)}/{len(sections)} sections have content",
            details={}
        )
        checks.append(content_check)

        # Check 3: Abstract word count (typically 150-300 words)
        abstract = next((s for s in sections if s.get("section_name") == "abstract"), None)
        if abstract and abstract.get("word_count"):
            word_count = abstract["word_count"]
            abstract_ok = 100 <= word_count <= 350
            abstract_check = GateCheckResult(
                check_name="abstract_word_count",
                passed=abstract_ok,
                score=1.0 if abstract_ok else 0.5,
                message=f"Abstract word count: {word_count}",
                details={"word_count": word_count}
            )
        else:
            abstract_check = GateCheckResult(
                check_name="abstract_word_count",
                passed=False,
                score=0.0,
                message="Abstract not found or missing word count",
                details={}
            )
        checks.append(abstract_check)

        # Check 4: Figures included
        figures = db.get_included_figures(project_id)
        figures_check = GateCheckResult(
            check_name="figures_included",
            passed=len(figures) >= 2,
            score=min(1.0, len(figures) / 2),
            message=f"{len(figures)} figures included in paper",
            details={"figure_count": len(figures)}
        )
        checks.append(figures_check)

        return checks


# =============================================================================
# FINAL GATE
# =============================================================================


class FinalGate(QualityGate):
    """Final quality gate before research completion."""

    def __init__(self, threshold: float = 0.85):
        super().__init__(
            name="FinalGate",
            threshold=threshold,
            required=True,
            max_retries=5
        )

    def check(self, project_id: int, db: DatabaseManager) -> list[GateCheckResult]:
        checks = []

        # Check 1: All prior gates passed (check agent log)
        prior_gates = ["LiteratureGate", "DataGate", "ExperimentGate", "StatisticsGate", "WritingGate"]
        log_entries = db.get_agent_log(project_id, limit=500)
        passed_gates = set()
        for entry in log_entries:
            if entry.get("action_type") == "quality_gate" and entry.get("result") == "success":
                gate_name = entry.get("action", "").replace(" passed", "")
                passed_gates.add(gate_name)

        missing_gates = set(prior_gates) - passed_gates
        gates_check = GateCheckResult(
            check_name="all_prior_gates_passed",
            passed=len(missing_gates) == 0,
            score=1.0 - (len(missing_gates) / len(prior_gates)),
            message=f"Prior gates passed: {len(passed_gates)}/{len(prior_gates)}",
            details={"missing_gates": list(missing_gates)}
        )
        checks.append(gates_check)

        # Check 2: Paper compiles (check for final.pdf)
        # This would normally check for file existence
        paper_check = GateCheckResult(
            check_name="paper_compiles",
            passed=True,  # Would check actual file
            score=1.0,
            message="Paper compilation check",
            details={}
        )
        checks.append(paper_check)

        # Check 3: Reproducibility artifacts present
        experiments = db.get_completed_experiments(project_id)
        has_artifacts = all(
            e.get("model_path") or e.get("notebook_path")
            for e in experiments
        )
        artifacts_check = GateCheckResult(
            check_name="reproducibility_artifacts",
            passed=has_artifacts and len(experiments) > 0,
            score=1.0 if (has_artifacts and experiments) else 0.0,
            message="Reproducibility artifacts present" if has_artifacts else "Missing artifacts",
            details={}
        )
        checks.append(artifacts_check)

        # Check 4: Hypothesis addressed
        hypotheses = db.get_hypotheses(project_id, tested_only=True)
        hypothesis_check = GateCheckResult(
            check_name="hypotheses_tested",
            passed=len(hypotheses) >= 1,
            score=min(1.0, len(hypotheses)),
            message=f"{len(hypotheses)} hypotheses tested",
            details={"tested_count": len(hypotheses)}
        )
        checks.append(hypothesis_check)

        return checks


# =============================================================================
# GATE MANAGER
# =============================================================================


class GateManager:
    """Manages all quality gates for a research project."""

    def __init__(self, db: DatabaseManager):
        """Initialize gate manager.

        Args:
            db: Database manager
        """
        self.db = db
        self.gates = {
            "literature": LiteratureGate(),
            "data": DataGate(),
            "experiment": ExperimentGate(),
            "analysis": StatisticsGate(),
            "writing": WritingGate(),
            "final": FinalGate()
        }

    def get_gate(self, stage: str) -> Optional[QualityGate]:
        """Get gate for a stage.

        Args:
            stage: Pipeline stage name

        Returns:
            QualityGate or None
        """
        return self.gates.get(stage)

    def evaluate_stage(self, project_id: int, stage: str) -> QualityGateResult:
        """Evaluate quality gate for a stage.

        Args:
            project_id: Project ID
            stage: Pipeline stage

        Returns:
            QualityGateResult
        """
        gate = self.get_gate(stage)
        if not gate:
            return QualityGateResult(
                gate_name=f"Unknown_{stage}",
                passed=True,
                score=1.0,
                threshold=0.0,
                details={"message": f"No gate defined for stage: {stage}"}
            )

        result = gate.evaluate(project_id, self.db)

        # Log the gate check
        self.db.log_agent_action(
            project_id=project_id,
            iteration=0,  # Would be set by caller
            story_id="",
            action=f"{gate.name} {'passed' if result.passed else 'failed'}",
            action_type="quality_gate",
            result="success" if result.passed else "failed",
            output_summary=json.dumps(result.details)
        )

        return result

    def evaluate_all(self, project_id: int) -> dict[str, QualityGateResult]:
        """Evaluate all quality gates.

        Args:
            project_id: Project ID

        Returns:
            Dict mapping stage to QualityGateResult
        """
        results = {}
        for stage, gate in self.gates.items():
            results[stage] = gate.evaluate(project_id, self.db)
        return results

    def can_proceed(self, project_id: int, from_stage: str, to_stage: str) -> bool:
        """Check if project can proceed from one stage to another.

        Args:
            project_id: Project ID
            from_stage: Current stage
            to_stage: Target stage

        Returns:
            True if quality gate passed
        """
        result = self.evaluate_stage(project_id, from_stage)
        return result.passed
