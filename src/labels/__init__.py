"""Target and ranking label construction package."""

from src.labels.targets import (
    TARGET_COL,
    construct_Targets_from_prices,
    compute_cross_sectional_percentile_ranks,
    fit_training_relevance_bins,
    assign_relevance_grades,
    verify_higher_label_means_better_return,
    extract_query_groups,
    build_pairwise_preferences_for_month,
    construct_all_targets,
    summarize_target_constructions,
)

__all__ = [
    "TARGET_COL",
    "construct_Targets_from_prices",
    "compute_cross_sectional_percentile_ranks",
    "fit_training_relevance_bins",
    "assign_relevance_grades",
    "verify_higher_label_means_better_return",
    "extract_query_groups",
    "build_pairwise_preferences_for_month",
    "construct_all_targets",
    "summarize_target_constructions",
]
