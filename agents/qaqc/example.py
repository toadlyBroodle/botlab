#!/usr/bin/env python3
import argparse
from qaqc.main import initialize

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example of using the QAQC agent")
    
    parser.add_argument("--query", type=str, default="Explain the impact of climate change on biodiversity",
                        help="Original query for context")
    parser.add_argument("--enable-telemetry", action="store_true", 
                        help="Whether to enable OpenTelemetry tracing")
    parser.add_argument("--output-selected-only", action="store_true",
                        help="Only output the selected text, not the analysis")
    
    return parser.parse_args()

def main():
    """Main entry point for the QAQC agent example."""
    args = parse_args()
    
    # Sample outputs to compare
    output1 = """
Climate change is having profound effects on biodiversity worldwide. Rising temperatures are altering habitats faster than many species can adapt, leading to range shifts, phenological changes, and in some cases, extinction. Ocean acidification, a direct result of increased atmospheric CO2, is threatening marine ecosystems, particularly coral reefs which support approximately 25% of all marine species.

Key impacts include:
1. Range shifts: Species are moving poleward and to higher elevations as their traditional habitats become unsuitable.
2. Phenological changes: Timing of seasonal activities (flowering, migration, breeding) is shifting, creating mismatches between interdependent species.
3. Physiological stress: Many organisms are experiencing stress from conditions exceeding their thermal tolerances.
4. Increased extinction risk: Species with limited dispersal ability or specialized habitat requirements face heightened extinction threats.

Ecosystem-level impacts are also significant, with changes in community composition, trophic relationships, and ecosystem services. For example, coral bleaching events are becoming more frequent and severe, threatening these biodiversity hotspots.

Conservation strategies must now incorporate climate change considerations, including protected area planning, assisted migration, and maintaining landscape connectivity to facilitate natural range shifts.
"""

    output2 = """
Climate change is affecting biodiversity in many ways. As temperatures rise, animals and plants are moving to new areas. Some species are having trouble adapting quickly enough.

The main effects include:
- Changes in where species can live
- Different timing for migration and breeding
- Some species dying out completely
- Problems with food chains when some species move or decline

The oceans are also changing because of climate change. They're getting warmer and more acidic, which is bad for coral reefs and the animals that live there.

Scientists are trying to help by creating protected areas and sometimes even moving species to new locations where they might survive better. They're also trying to connect natural areas so animals can move between them as the climate changes.

Overall, climate change is one of the biggest threats to biodiversity today, along with habitat loss and pollution.
"""

    # Create outputs dictionary
    outputs = {
        "Detailed Scientific Response": output1,
        "Simplified General Response": output2
    }

    # Initialize the QAQC agent
    compare_outputs = initialize(
        enable_telemetry=args.enable_telemetry
    )
    
    # Run the comparison
    selected_output, analysis, selected_name = compare_outputs(args.query, outputs)
    
    # Print the result
    if args.output_selected_only:
        print(selected_output)
    else:
        print("\n=== QAQC Agent Comparison Result ===\n")
        print(analysis)
        print("\n=== Selected Output ===\n")
        print(f"Selected: {selected_name}")
        print(selected_output)

if __name__ == "__main__":
    main() 