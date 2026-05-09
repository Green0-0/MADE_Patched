import logging
import os

from . import filters, generators, planners, scorers
from .base import Agent, FilterResult, ScoreResult, WorkflowAgent
from .llm_react_orchestrator import LLMReActOrchestratorAgent
from .workflow import OneShotWorkflowAgent

logger = logging.getLogger(__name__)

# Initialize Langfuse DSPy tracing if environment variables are present
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    try:
        import langfuse
        from openinference.instrumentation.dspy import DSPyInstrumentor

        DSPyInstrumentor().instrument()
        logger.info("Langfuse DSPy instrumentation enabled.")
    except ImportError:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY is set but langfuse or openinference-instrumentation-dspy "
            "is not installed. Tracing will not be enabled."
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse instrumentation: {e}")

__all__ = [
    "Agent",
    "WorkflowAgent",
    "FilterResult",
    "ScoreResult",
    "LLMReActOrchestratorAgent",
    "OneShotWorkflowAgent",
    "filters",
    "planners",
    "generators",
    "scorers",
]
