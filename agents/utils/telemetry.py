from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def start_telemetry():
    # Create and set the tracer provider
    trace_provider = TracerProvider()
    
    # Add console exporter for immediate feedback
    console_exporter = ConsoleSpanExporter()
    trace_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
    
    # Create OTLP exporter pointing to Jaeger
    endpoint = "http://0.0.0.0:4317"
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the trace provider as the global default
    trace.set_tracer_provider(trace_provider)
    
    # Register the instrumentation
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider, skip_dep_check=True)
    
    return trace_provider

if __name__ == "__main__":
    start_telemetry()
