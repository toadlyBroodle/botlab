from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def start_telemetry():
    """Initialize OpenTelemetry with OTLP exporter for Phoenix"""
    # Create and configure the OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://0.0.0.0:6006/v1/traces",
        headers={"Content-Type": "application/x-protobuf"}
    )
    
    # Set up the trace provider with batch processor
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the trace provider as the global default
    trace.set_tracer_provider(trace_provider)
    
    # Instrument smolagents with our trace provider
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

if __name__ == "__main__":
    start_telemetry()
