"""Tab rendering modules for the Electricity Dashboard."""
from .overview import render_overview
from .hourly import render_hourly
from .trends import render_trends
from .clustering import render_clustering
from .discounts import render_discounts
from .calculator import render_calculator
from .weather import render_weather
from .report import render_report
from .about import render_about

__all__ = [
    "render_overview", "render_hourly", "render_trends", "render_clustering",
    "render_discounts", "render_calculator", "render_weather",
    "render_report", "render_about",
]
