"""PRD Generator for FRINK.

This module generates complete PRDs (Product Requirements Documents) from
research topic definitions. The generated PRD contains all user stories
needed to complete autonomous research.
"""

from datetime import datetime
from typing import Optional

from lib.schemas import (
    PRDMetadata,
    ResearchPRD,
    ResearchTopic,
    UserStory,
)


# =============================================================================
# STORY TEMPLATES BY STAGE
# =============================================================================

LITERATURE_STORIES = [
    {
        "id": "LIT-001",
        "title": "Define search strategy and keywords",
        "description": "Develop comprehensive search strategy based on research topic, including databases to search and keyword combinations",
        "skills_required": ["literature-review"],
        "acceptance_criteria": [
            "Search keywords defined",
            "Target databases identified",
            "Search strategy documented"
        ],
        "priority": 1,
        "outputs": ["search_strategy.json"],
        "estimated_duration_minutes": 15
    },
    {
        "id": "LIT-002",
        "title": "Search academic databases",
        "description": "Execute search strategy across OpenAlex, Semantic Scholar, and other relevant databases",
        "skills_required": ["literature-review", "semantic-scholar", "openalex"],
        "acceptance_criteria": [
            "At least 50 papers retrieved",
            "Results deduplicated",
            "Metadata extracted and stored"
        ],
        "priority": 2,
        "dependencies": ["LIT-001"],
        "outputs": ["raw_literature.json"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "LIT-003",
        "title": "Screen papers by title and abstract",
        "description": "Apply inclusion/exclusion criteria to filter papers based on title and abstract relevance",
        "skills_required": ["literature-review"],
        "acceptance_criteria": [
            "All papers screened",
            "Relevance scores assigned",
            "At least 20 papers pass abstract screening"
        ],
        "priority": 3,
        "dependencies": ["LIT-002"],
        "outputs": ["screened_literature.json"],
        "estimated_duration_minutes": 45
    },
    {
        "id": "LIT-004",
        "title": "Extract key findings from included papers",
        "description": "Extract methods, results, and key findings from papers that passed screening",
        "skills_required": ["literature-review", "pdf-extraction"],
        "acceptance_criteria": [
            "Key findings extracted from at least 15 papers",
            "Methods documented",
            "Gaps identified"
        ],
        "priority": 4,
        "dependencies": ["LIT-003"],
        "outputs": ["literature_synthesis.json", "research_gaps.md"],
        "estimated_duration_minutes": 60
    },
    {
        "id": "LIT-005",
        "title": "Generate literature review summary",
        "description": "Synthesize findings into a coherent literature review narrative",
        "skills_required": ["literature-review", "academic-writing"],
        "acceptance_criteria": [
            "Literature review draft generated",
            "Key themes identified",
            "Research gap clearly articulated"
        ],
        "priority": 5,
        "dependencies": ["LIT-004"],
        "outputs": ["literature_review_draft.md"],
        "estimated_duration_minutes": 45
    }
]

HYPOTHESIS_STORIES = [
    {
        "id": "HYP-001",
        "title": "Refine primary hypothesis",
        "description": "Refine the initial hypothesis based on literature review findings",
        "skills_required": ["hypogenic", "scientific-reasoning"],
        "acceptance_criteria": [
            "Hypothesis refined based on literature",
            "Hypothesis is testable and falsifiable",
            "Operationalization defined"
        ],
        "priority": 1,
        "dependencies": ["LIT-005"],
        "outputs": ["refined_hypothesis.json"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "HYP-002",
        "title": "Generate secondary hypotheses",
        "description": "Generate secondary and exploratory hypotheses based on research gaps",
        "skills_required": ["hypogenic"],
        "acceptance_criteria": [
            "At least 2 secondary hypotheses generated",
            "All hypotheses are testable",
            "Priority ranking established"
        ],
        "priority": 2,
        "dependencies": ["HYP-001"],
        "outputs": ["all_hypotheses.json"],
        "estimated_duration_minutes": 20
    }
]

DATA_STORIES = [
    {
        "id": "DATA-001",
        "title": "Download and validate dataset",
        "description": "Download dataset from specified source and validate integrity",
        "skills_required": ["kaggle-api", "data-validation"],
        "acceptance_criteria": [
            "Dataset downloaded successfully",
            "Data integrity verified",
            "Schema documented"
        ],
        "priority": 1,
        "dependencies": ["HYP-002"],
        "outputs": ["raw_data/", "data_manifest.json"],
        "estimated_duration_minutes": 20
    },
    {
        "id": "DATA-002",
        "title": "Perform exploratory data analysis",
        "description": "Comprehensive EDA including distributions, correlations, and quality assessment",
        "skills_required": ["pandas-expert", "eda", "matplotlib"],
        "acceptance_criteria": [
            "EDA report generated",
            "Missing value analysis complete",
            "Feature distributions documented",
            "Outliers identified"
        ],
        "priority": 2,
        "dependencies": ["DATA-001"],
        "outputs": ["eda_report.html", "eda_figures/"],
        "estimated_duration_minutes": 45
    },
    {
        "id": "DATA-003",
        "title": "Preprocess and clean data",
        "description": "Apply preprocessing including handling missing values, encoding, and scaling",
        "skills_required": ["pandas-expert", "scikit-learn", "feature-engineering"],
        "acceptance_criteria": [
            "Missing values handled appropriately",
            "Categorical variables encoded",
            "Numerical features scaled",
            "Preprocessing pipeline saved"
        ],
        "priority": 3,
        "dependencies": ["DATA-002"],
        "outputs": ["processed_data/", "preprocessing_pipeline.pkl"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "DATA-004",
        "title": "Create train/validation/test splits",
        "description": "Split data with appropriate stratification and document split ratios",
        "skills_required": ["scikit-learn"],
        "acceptance_criteria": [
            "Data split with reproducible seed",
            "Stratification applied if classification",
            "No data leakage between splits",
            "Split statistics documented"
        ],
        "priority": 4,
        "dependencies": ["DATA-003"],
        "outputs": ["train.csv", "val.csv", "test.csv", "split_info.json"],
        "estimated_duration_minutes": 15
    }
]

EXPERIMENT_STORIES = [
    {
        "id": "EXP-001",
        "title": "Implement simple baseline",
        "description": "Implement simple baseline model (e.g., majority class, mean prediction)",
        "skills_required": ["scikit-learn"],
        "acceptance_criteria": [
            "Simple baseline implemented",
            "Results on validation set recorded",
            "Baseline serves as lower bound"
        ],
        "priority": 1,
        "dependencies": ["DATA-004"],
        "outputs": ["baseline_simple_results.json"],
        "estimated_duration_minutes": 20
    },
    {
        "id": "EXP-002",
        "title": "Implement standard ML baselines",
        "description": "Implement 2-3 standard ML baselines (e.g., Random Forest, XGBoost, SVM)",
        "skills_required": ["scikit-learn", "xgboost"],
        "acceptance_criteria": [
            "At least 2 standard baselines implemented",
            "Cross-validation performed",
            "Results documented"
        ],
        "priority": 2,
        "dependencies": ["EXP-001"],
        "outputs": ["baseline_ml_results.json", "models/baselines/"],
        "estimated_duration_minutes": 45
    },
    {
        "id": "EXP-003",
        "title": "Implement proposed method",
        "description": "Implement the proposed method based on the research hypothesis",
        "skills_required": ["pytorch", "scikit-learn", "model-development"],
        "acceptance_criteria": [
            "Proposed method implemented",
            "Method follows hypothesis",
            "Code is reproducible"
        ],
        "priority": 3,
        "dependencies": ["EXP-002"],
        "outputs": ["proposed_method.py", "models/proposed/"],
        "estimated_duration_minutes": 90
    },
    {
        "id": "EXP-004",
        "title": "Hyperparameter tuning",
        "description": "Tune hyperparameters for proposed method and baselines",
        "skills_required": ["optuna", "hyperparameter-tuning"],
        "acceptance_criteria": [
            "Hyperparameter search completed",
            "Best parameters documented",
            "Search history saved"
        ],
        "priority": 4,
        "dependencies": ["EXP-003"],
        "outputs": ["tuning_results.json", "best_params.json"],
        "estimated_duration_minutes": 60
    },
    {
        "id": "EXP-005",
        "title": "Run final evaluation on test set",
        "description": "Evaluate all models on held-out test set with final hyperparameters",
        "skills_required": ["scikit-learn", "evaluation-metrics"],
        "acceptance_criteria": [
            "All models evaluated on test set",
            "Multiple metrics reported",
            "Confidence intervals calculated"
        ],
        "priority": 5,
        "dependencies": ["EXP-004"],
        "outputs": ["final_results.json", "test_predictions/"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "EXP-006",
        "title": "Perform ablation studies",
        "description": "Conduct ablation studies to understand contribution of each component",
        "skills_required": ["ablation-study"],
        "acceptance_criteria": [
            "Key components identified for ablation",
            "Each ablation evaluated",
            "Contribution of each component quantified"
        ],
        "priority": 6,
        "dependencies": ["EXP-005"],
        "outputs": ["ablation_results.json"],
        "estimated_duration_minutes": 60
    }
]

ANALYSIS_STORIES = [
    {
        "id": "STAT-001",
        "title": "Perform statistical significance tests",
        "description": "Test statistical significance of improvements over baselines",
        "skills_required": ["statistical-analysis", "scipy"],
        "acceptance_criteria": [
            "Appropriate test selected (paired t-test, Wilcoxon, etc.)",
            "p-values calculated",
            "Multiple comparison correction applied"
        ],
        "priority": 1,
        "dependencies": ["EXP-005"],
        "outputs": ["significance_tests.json"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "STAT-002",
        "title": "Calculate effect sizes",
        "description": "Calculate and interpret effect sizes (Cohen's d, etc.)",
        "skills_required": ["statistical-analysis"],
        "acceptance_criteria": [
            "Effect sizes calculated",
            "Confidence intervals computed",
            "Practical significance assessed"
        ],
        "priority": 2,
        "dependencies": ["STAT-001"],
        "outputs": ["effect_sizes.json"],
        "estimated_duration_minutes": 20
    },
    {
        "id": "STAT-003",
        "title": "Analyze errors and failure cases",
        "description": "Systematic analysis of model errors and failure modes",
        "skills_required": ["error-analysis", "pandas-expert"],
        "acceptance_criteria": [
            "Error patterns identified",
            "Failure cases categorized",
            "Insights documented"
        ],
        "priority": 3,
        "dependencies": ["EXP-005"],
        "outputs": ["error_analysis.json", "failure_cases.csv"],
        "estimated_duration_minutes": 45
    }
]

VISUALIZATION_STORIES = [
    {
        "id": "VIZ-001",
        "title": "Create results comparison plots",
        "description": "Generate publication-quality plots comparing model performance",
        "skills_required": ["matplotlib", "seaborn", "publication-figures"],
        "acceptance_criteria": [
            "Comparison bar/box plots generated",
            "Confidence intervals shown",
            "Publication quality (300 DPI)"
        ],
        "priority": 1,
        "dependencies": ["STAT-002"],
        "outputs": ["figures/results_comparison.pdf"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "VIZ-002",
        "title": "Create ablation study plots",
        "description": "Visualize ablation study results",
        "skills_required": ["matplotlib", "seaborn"],
        "acceptance_criteria": [
            "Ablation results visualized",
            "Component contributions clear",
            "Publication quality"
        ],
        "priority": 2,
        "dependencies": ["EXP-006", "VIZ-001"],
        "outputs": ["figures/ablation.pdf"],
        "estimated_duration_minutes": 20
    },
    {
        "id": "VIZ-003",
        "title": "Create method architecture diagram",
        "description": "Create diagram illustrating the proposed method architecture",
        "skills_required": ["diagram-generation"],
        "acceptance_criteria": [
            "Architecture clearly illustrated",
            "Key components labeled",
            "Suitable for paper"
        ],
        "priority": 3,
        "dependencies": ["EXP-003"],
        "outputs": ["figures/architecture.pdf"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "VIZ-004",
        "title": "Generate results tables",
        "description": "Generate LaTeX tables for main results and ablations",
        "skills_required": ["latex-tables"],
        "acceptance_criteria": [
            "Main results table generated",
            "Ablation table generated",
            "Best results highlighted"
        ],
        "priority": 4,
        "dependencies": ["STAT-002", "EXP-006"],
        "outputs": ["tables/main_results.tex", "tables/ablation.tex"],
        "estimated_duration_minutes": 25
    }
]

WRITING_STORIES = [
    {
        "id": "WRITE-001",
        "title": "Write abstract",
        "description": "Write concise abstract summarizing the research",
        "skills_required": ["academic-writing", "abstract-writing"],
        "acceptance_criteria": [
            "Abstract within word limit",
            "All key elements included (background, method, results, conclusion)",
            "Clear and concise"
        ],
        "priority": 1,
        "dependencies": ["VIZ-001", "STAT-002"],
        "outputs": ["paper/abstract.tex"],
        "estimated_duration_minutes": 30
    },
    {
        "id": "WRITE-002",
        "title": "Write introduction",
        "description": "Write introduction section with motivation and contributions",
        "skills_required": ["academic-writing"],
        "acceptance_criteria": [
            "Motivation clearly established",
            "Research gap identified",
            "Contributions listed",
            "Paper structure outlined"
        ],
        "priority": 2,
        "dependencies": ["LIT-005", "WRITE-001"],
        "outputs": ["paper/introduction.tex"],
        "estimated_duration_minutes": 60
    },
    {
        "id": "WRITE-003",
        "title": "Write related work section",
        "description": "Write related work section based on literature review",
        "skills_required": ["academic-writing", "literature-review"],
        "acceptance_criteria": [
            "Key related work discussed",
            "Comparison to our approach",
            "Properly cited"
        ],
        "priority": 3,
        "dependencies": ["LIT-005", "WRITE-002"],
        "outputs": ["paper/related_work.tex"],
        "estimated_duration_minutes": 45
    },
    {
        "id": "WRITE-004",
        "title": "Write methodology section",
        "description": "Write detailed methodology section describing the proposed approach",
        "skills_required": ["academic-writing", "technical-writing"],
        "acceptance_criteria": [
            "Method clearly explained",
            "Notation consistent",
            "Reproducible from description"
        ],
        "priority": 4,
        "dependencies": ["EXP-003", "WRITE-003"],
        "outputs": ["paper/methodology.tex"],
        "estimated_duration_minutes": 90
    },
    {
        "id": "WRITE-005",
        "title": "Write experiments section",
        "description": "Write experiments section including setup, baselines, and results",
        "skills_required": ["academic-writing"],
        "acceptance_criteria": [
            "Experimental setup detailed",
            "Baselines described",
            "Results presented clearly"
        ],
        "priority": 5,
        "dependencies": ["VIZ-004", "WRITE-004"],
        "outputs": ["paper/experiments.tex"],
        "estimated_duration_minutes": 75
    },
    {
        "id": "WRITE-006",
        "title": "Write discussion and conclusion",
        "description": "Write discussion of results and conclusion",
        "skills_required": ["academic-writing"],
        "acceptance_criteria": [
            "Results interpreted",
            "Limitations discussed",
            "Future work suggested",
            "Conclusions drawn"
        ],
        "priority": 6,
        "dependencies": ["WRITE-005"],
        "outputs": ["paper/conclusion.tex"],
        "estimated_duration_minutes": 45
    }
]

REVIEW_STORIES = [
    {
        "id": "REVIEW-001",
        "title": "Self-review for technical correctness",
        "description": "Review paper for technical errors and inconsistencies",
        "skills_required": ["peer-review", "technical-review"],
        "acceptance_criteria": [
            "Technical claims verified",
            "Math checked",
            "No inconsistencies"
        ],
        "priority": 1,
        "dependencies": ["WRITE-006"],
        "outputs": ["review/technical_review.md"],
        "estimated_duration_minutes": 60
    },
    {
        "id": "REVIEW-002",
        "title": "Verify reproducibility",
        "description": "Verify all experiments are reproducible from paper description",
        "skills_required": ["reproducibility-check"],
        "acceptance_criteria": [
            "All hyperparameters documented",
            "Code matches paper",
            "Results reproducible"
        ],
        "priority": 2,
        "dependencies": ["REVIEW-001"],
        "outputs": ["review/reproducibility_check.md"],
        "estimated_duration_minutes": 45
    },
    {
        "id": "REVIEW-003",
        "title": "Final paper compilation",
        "description": "Compile final paper and verify all components",
        "skills_required": ["latex", "paper-compilation"],
        "acceptance_criteria": [
            "Paper compiles without errors",
            "All figures included",
            "Bibliography complete",
            "Within page limit"
        ],
        "priority": 3,
        "dependencies": ["REVIEW-002"],
        "outputs": ["paper/final.pdf", "paper/final.tex"],
        "estimated_duration_minutes": 30
    }
]


# =============================================================================
# PRD GENERATOR
# =============================================================================


class PRDGenerator:
    """Generates complete PRDs from research topics."""

    def __init__(self, topic: ResearchTopic):
        """Initialize generator with research topic.

        Args:
            topic: Research topic definition
        """
        self.topic = topic

    def generate(self, project_name: str) -> ResearchPRD:
        """Generate a complete PRD.

        Args:
            project_name: Name for the research project

        Returns:
            Complete ResearchPRD
        """
        branch_name = self._create_branch_name(project_name)
        stories = self._generate_all_stories()

        return ResearchPRD(
            project=project_name,
            branchName=branch_name,
            topic=self.topic,
            userStories=stories,
            metadata=PRDMetadata(
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                version="1.0.0",
                iteration_count=0
            )
        )

    def _create_branch_name(self, project_name: str) -> str:
        """Create git branch name from project name."""
        return project_name.lower().replace(" ", "-").replace("_", "-")[:50]

    def _generate_all_stories(self) -> list[UserStory]:
        """Generate all user stories for the research pipeline."""
        stories = []

        # Add stories from each stage
        stories.extend(self._create_stories_from_templates(LITERATURE_STORIES, "literature"))
        stories.extend(self._create_stories_from_templates(HYPOTHESIS_STORIES, "hypothesis"))
        stories.extend(self._create_stories_from_templates(DATA_STORIES, "data"))
        stories.extend(self._create_stories_from_templates(EXPERIMENT_STORIES, "experiment"))
        stories.extend(self._create_stories_from_templates(ANALYSIS_STORIES, "analysis"))
        stories.extend(self._create_stories_from_templates(VISUALIZATION_STORIES, "visualization"))
        stories.extend(self._create_stories_from_templates(WRITING_STORIES, "writing"))
        stories.extend(self._create_stories_from_templates(REVIEW_STORIES, "review"))

        # Customize based on topic
        stories = self._customize_for_topic(stories)

        return stories

    def _create_stories_from_templates(
        self,
        templates: list[dict],
        stage: str
    ) -> list[UserStory]:
        """Create UserStory objects from templates."""
        stories = []
        for template in templates:
            story = UserStory(
                id=template["id"],
                title=template["title"],
                description=template.get("description", ""),
                stage=stage,
                skills_required=template.get("skills_required", []),
                acceptanceCriteria=template.get("acceptance_criteria", []),
                priority=template.get("priority", 1),
                dependencies=template.get("dependencies", []),
                outputs=template.get("outputs", []),
                estimated_duration_minutes=template.get("estimated_duration_minutes", 30)
            )
            stories.append(story)
        return stories

    def _customize_for_topic(self, stories: list[UserStory]) -> list[UserStory]:
        """Customize stories based on research topic.

        This method modifies stories based on:
        - Domain-specific requirements
        - Dataset constraints
        - Method constraints
        """
        # Add domain-specific skills
        domain_skills = self._get_domain_skills()

        for story in stories:
            # Add domain skills to relevant stories
            if story.stage in ("experiment", "data", "analysis"):
                story.skills_required = list(
                    set(story.skills_required) | set(domain_skills)
                )

        # If GPU not required, adjust experiment stories
        if not self.topic.constraints.gpu_required:
            for story in stories:
                if "pytorch" in story.skills_required:
                    story.notes = "CPU-only execution"

        # Add method-specific stories if required methods specified
        if self.topic.constraints.required_methods:
            # Could add additional experiment stories here
            pass

        return stories

    def _get_domain_skills(self) -> list[str]:
        """Get domain-specific skills to add to stories."""
        domain_skills = {
            "ML": ["scikit-learn", "pytorch"],
            "BIOINFORMATICS": ["bioinformatics", "biopython"],
            "STATISTICS": ["statistical-analysis", "scipy", "statsmodels"],
            "CHEMISTRY": ["rdkit", "chemistry"],
            "PHYSICS": ["physics", "numpy"],
            "MEDICINE": ["medical-stats", "survival-analysis"],
            "SOCIAL_SCIENCE": ["social-science-stats", "survey-analysis"],
            "COMPUTER_VISION": ["pytorch", "torchvision", "opencv"],
            "NLP": ["transformers", "nltk", "spacy"]
        }
        return domain_skills.get(self.topic.domain, [])


def generate_prd(
    project_name: str,
    topic: ResearchTopic,
    output_path: Optional[str] = None
) -> ResearchPRD:
    """Generate a complete PRD for a research project.

    Args:
        project_name: Name for the research project
        topic: Research topic definition
        output_path: Optional path to save PRD JSON

    Returns:
        Generated ResearchPRD
    """
    generator = PRDGenerator(topic)
    prd = generator.generate(project_name)

    if output_path:
        prd.to_file(output_path)

    return prd
