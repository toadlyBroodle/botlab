from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import functools
import inspect
import os

# Global variable to track the current agent context
_CURRENT_AGENT_CONTEXT = {
    "name": None,
    "type": None
}

def suppress_litellm_logs():
    """Suppress INFO-level logs from LiteLLM by setting its logger level to ERROR.
    
    This function should be called at the start of your application to prevent
    LiteLLM from printing INFO logs like "LiteLLM completion() model= gemini-2.0-flash; provider = gemini"
    """
    import logging
    
    # Set LiteLLM logger to ERROR level to suppress INFO logs
    logging.getLogger("litellm").setLevel(logging.ERROR)
    
    # Disable LiteLLM's internal logging
    try:
        import litellm
        litellm.utils.logging_enabled = False
        os.environ["LITELLM_LOG_VERBOSE"] = "false"
    except ImportError:
        pass  # LiteLLM not installed, nothing to do

def start_telemetry(agent_name=None, agent_type=None):
    """Initialize OpenTelemetry with OTLP exporter for Phoenix
    
    Args:
        agent_name: Optional name of the specific agent instance
        agent_type: Optional type of agent (researcher, writer, editor, etc.)
        
    Returns:
        A tracer that can be used for custom spans
    """
    global _CURRENT_AGENT_CONTEXT
    
    # Store agent context for later use
    if agent_name:
        _CURRENT_AGENT_CONTEXT["name"] = agent_name
    if agent_type:
        _CURRENT_AGENT_CONTEXT["type"] = agent_type
    
    # Create a resource with service name and other attributes
    resource_attributes = {
        ResourceAttributes.SERVICE_NAME: "smolagents-service",
        ResourceAttributes.SERVICE_VERSION: "0.1.0",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
    }
    
    # Add agent-specific attributes if provided
    if agent_name:
        resource_attributes["agent.name"] = agent_name
    if agent_type:
        resource_attributes["agent.type"] = agent_type
    
    resource = Resource.create(resource_attributes)
    
    # Create and configure the OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://0.0.0.0:6006/v1/traces",
        headers={"Content-Type": "application/x-protobuf"}
    )
    
    # Set up the trace provider with batch processor and resource
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the trace provider as the global default
    trace.set_tracer_provider(trace_provider)
    
    # Configure SmolagentsInstrumentor with custom span attributes
    def add_agent_context(span, agent_context=None):
        """Add agent context to spans created by smolagents"""
        if not agent_context:
            agent_context = _CURRENT_AGENT_CONTEXT
            
        if agent_context.get("name"):
            span.set_attribute("agent.name", agent_context["name"])
        if agent_context.get("type"):
            span.set_attribute("agent.type", agent_context["type"])
    
    # Instrument smolagents with our trace provider and custom span processor
    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=trace_provider,
        span_callback=add_agent_context
    )
    
    # Return a tracer that can be used for custom spans
    return trace.get_tracer(agent_name or "smolagents")

def traced(span_name=None, attributes=None):
    """Decorator to trace a function with a custom span name and attributes
    
    Args:
        span_name: Optional custom name for the span (defaults to function name)
        attributes: Optional dict of attributes to add to the span
        
    Returns:
        Decorated function with tracing
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the function name if span_name is not provided
            name = span_name or f"{func.__module__}.{func.__name__}"
            
            # Get the tracer
            tracer = trace.get_tracer(func.__module__)
            
            # Start a span
            with tracer.start_as_current_span(name) as span:
                # Add current agent context
                if _CURRENT_AGENT_CONTEXT.get("name"):
                    span.set_attribute("agent.name", _CURRENT_AGENT_CONTEXT["name"])
                if _CURRENT_AGENT_CONTEXT.get("type"):
                    span.set_attribute("agent.type", _CURRENT_AGENT_CONTEXT["type"])
                
                # Add attributes if provided
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function signature as attributes
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Add arguments as attributes (excluding self/cls)
                for param_name, param_value in bound_args.arguments.items():
                    if param_name not in ('self', 'cls'):
                        # Convert to string to avoid serialization issues
                        try:
                            if isinstance(param_value, (str, int, float, bool)):
                                span.set_attribute(f"arg.{param_name}", param_value)
                            else:
                                span.set_attribute(f"arg.{param_name}", str(type(param_value)))
                        except:
                            pass
                
                # Call the original function
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

if __name__ == "__main__":
    start_telemetry()
