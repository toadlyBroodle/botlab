"""Newsletter pipeline package.

Exports the `NewsletterPipeline` for orchestrating Researcher → Writer → Editor.
"""

from .agents import NewsletterPipeline

__all__ = ["NewsletterPipeline"]


