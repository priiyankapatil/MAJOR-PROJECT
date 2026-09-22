# Component Layout

This package groups the current pipeline into logical components while keeping
the existing top-level scripts intact.

## Components

- `ingestion`: PDF extraction, cleaning, chunking, and storage
- `retrieval`: dense + sparse retrieval wrappers
- `routing`: query classification and entropy-based path selection
- `context`: semantic bridge, temporal filter, phenology gate, weather enrichment
- `generation`: answer generation wrappers
- `explainability`: sentence provenance reporting
- `safety`: regulatory compliance and updater logic
- `evaluation`: baseline comparison and metrics scaffold

## Principle

The runtime pipeline still uses the existing modules directly. These wrappers
provide a cleaner component boundary for explanation, future refactoring, and
evaluation work.
