# BigQuery Cost Intelligence Engine

The BigQuery Cost Intelligence Engine (BCIE) is a serverless application that analyzes BigQuery datasets to provide cost optimization recommendations with ROI estimates.

## Module Structure

### API Layer
The API layer handles webhook calls from Retool and orchestrates the analysis workflow.

### Analysis Modules
- **Metadata Extractor**: Extracts schema, size, and usage patterns
- **Storage Optimizer**: Analyzes partitioning, clustering options
- **Query Optimizer**: Identifies inefficient query patterns
- **Schema Optimizer**: Finds unused columns and type optimizations

### Recommendation Engine
Generates prioritized recommendations with implementation steps and ROI estimates.

### Implementation Plan Generator
Creates SQL scripts and step-by-step guides for implementing recommendations.

## Development

See the `architecture.md` file for the complete system design and component details.