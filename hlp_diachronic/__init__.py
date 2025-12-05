"""
HLP Diachronic - Diachronic Analysis Package

This package provides comprehensive support for diachronic linguistic
analysis, including periodization, change metrics, and visualization
data generation for tracking linguistic changes over time.

Modules:
    periodization: Define and manage historical periods
    change_metrics: Calculate diachronic change statistics
    visualization_data: Generate data for visualizations

University of Athens - Nikolaos Lavidas
"""

from hlp_diachronic.periodization import (
    PeriodDefinition,
    PeriodSystem,
    GreekPeriodSystem,
    LatinPeriodSystem,
    CustomPeriodSystem,
    get_period_for_date,
    get_period_for_text,
    create_period_system,
)

from hlp_diachronic.change_metrics import (
    ChangeMetric,
    ChangeType,
    DiachronicChange,
    ChangeCalculator,
    calculate_frequency_change,
    calculate_pattern_change,
    calculate_construction_change,
    detect_innovations,
    detect_losses,
)

from hlp_diachronic.visualization_data import (
    VisualizationData,
    TimeSeriesData,
    DistributionData,
    NetworkData,
    generate_timeline_data,
    generate_frequency_chart_data,
    generate_distribution_data,
    generate_network_data,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "PeriodDefinition",
    "PeriodSystem",
    "GreekPeriodSystem",
    "LatinPeriodSystem",
    "CustomPeriodSystem",
    "get_period_for_date",
    "get_period_for_text",
    "create_period_system",
    "ChangeMetric",
    "ChangeType",
    "DiachronicChange",
    "ChangeCalculator",
    "calculate_frequency_change",
    "calculate_pattern_change",
    "calculate_construction_change",
    "detect_innovations",
    "detect_losses",
    "VisualizationData",
    "TimeSeriesData",
    "DistributionData",
    "NetworkData",
    "generate_timeline_data",
    "generate_frequency_chart_data",
    "generate_distribution_data",
    "generate_network_data",
]
