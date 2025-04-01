# BigQuery Cost Intelligence Engine (BCIE) Architecture

## System Architecture Diagram

```
                                      +-------------------+
                                      |                   |
                                      |  Retool Dashboard |
                                      |                   |
                                      +--------+----------+
                                               |
                                               | Webhook Call
                                               v
+------------------+     +--------------+    +---------------+     +-------------------+
|                  |     |              |    |               |     |                   |
| Cloud Scheduler  +---->+ Cloud Pub/Sub+---->  Cloud Run    |     |  Cloud Functions  |
| (Batch Analysis) |     | (Event Bus)  |    | (API Gateway) +---->+  (Analysis Modules)|
|                  |     |              |    |               |     |                   |
+------------------+     +--------------+    +-------+-------+     +---------+---------+
                                                     |                       |
                                                     |                       |
                          +------------------------+ |                       |
                          |                        | |                       |
                          |    Secret Manager     <-+                       |
                          | (API Keys/Credentials)|                         |
                          |                        |                         |
                          +------------------------+                         |
                                                                            |
                                                                            v
                                                             +-------------+--------------+
                                                             |                            |
                                                             |      BigQuery              |
                                                             | (Data & Recommendations)   |
                                                             |                            |
                                                             +----------------------------+
                                                                         ^
                                                                         |
                                                             +-----------+------------+
                                                             |                        |
                                                             |     Cloud Storage      |
                                                             | (Processing Artifacts) |
                                                             |                        |
                                                             +------------------------+
```

## Component Descriptions

### Frontend Layer
- **Retool Dashboard**: Custom dashboard for initiating analysis, viewing recommendations, and tracking implementation status.

### API & Control Layer
- **Cloud Run (API Gateway)**: 
  - Exposes REST API endpoints for triggering analysis
  - Manages authentication and request validation
  - Orchestrates the analysis workflow
  - Provides status updates

### Processing & Analysis Layer
- **Cloud Pub/Sub**:
  - Decouples webhook requests from processing
  - Enables asynchronous, parallel processing
  - Manages workload distribution

- **Cloud Functions**:
  - **Metadata Extractor**: Queries INFORMATION_SCHEMA for table metadata
  - **Storage Optimizer**: Analyzes compression, partitioning and clustering options
  - **Query Optimizer**: Analyzes query patterns for optimization
  - **Schema Optimizer**: Identifies redundant columns and type optimization opportunities
  - **ROI Calculator**: Estimates cost savings and implementation effort
  - **Recommendation Formatter**: Structures recommendations for storage

### Data Storage Layer
- **BigQuery**:
  - Stores analyzed datasets
  - Stores recommendations and implementation details
  - Tracks historical cost patterns

- **Cloud Storage**:
  - Stores processing artifacts and intermediate results
  - Enables efficient data exchange between services

### Security Layer
- **Secret Manager**: Securely stores API keys and credentials

### Automation Layer
- **Cloud Scheduler**: Triggers batch analysis on schedule

## Core Modules

1. **Data Connector & Metadata Extractor**
   - Extracts schema, size, usage patterns from INFORMATION_SCHEMA
   - Performs deep analysis of dataset characteristics
   - Identifies table dependencies and relationships

2. **Analysis Modules**
   - **Storage Optimizer**: Analyzes partitioning, clustering, and compression options
   - **Query Optimizer**: Analyzes query patterns, identifies inefficient queries
   - **Schema Optimizer**: Identifies unused columns, suboptimal data types

3. **Recommendation Engine**
   - Prioritizes recommendations based on impact
   - Generates specific implementation steps
   - Calculates ROI and effort estimates

4. **Implementation Plan Generator**
   - Creates step-by-step implementation guides
   - Generates SQL scripts for implementing changes
   - Estimates timeline and resource requirements

5. **API Integration Layer**
   - Webhook handlers for Retool integration
   - Status tracking and notification system
   - Authentication and authorization

## Project Structure

```
bigquerycostopt/
├── api/
│   ├── endpoints.py  # Cloud Run API definitions
│   ├── auth.py       # Authentication middleware
│   └── status.py     # Analysis status tracking
├── analysis/
│   ├── metadata.py   # Metadata extraction functions
│   ├── storage.py    # Storage optimization analysis
│   ├── query.py      # Query pattern analysis
│   └── schema.py     # Schema optimization analysis
├── recommender/
│   ├── engine.py     # Recommendation generation
│   ├── roi.py        # ROI calculation logic
│   └── formatter.py  # Recommendation formatting
├── implementation/
│   ├── planner.py    # Implementation plan generation
│   └── scripts.py    # SQL script generators
├── connectors/
│   ├── bigquery.py   # BigQuery connection manager
│   └── pubsub.py     # Pub/Sub integration
├── utils/
│   ├── logging.py    # Logging utilities
│   └── security.py   # Security helper functions
└── tests/
    ├── unit/         # Unit tests for all modules
    └── integration/  # Integration tests
```

## Technology Stack

- **Languages**: Python 3.9+
- **Cloud Platform**: Google Cloud Platform (GCP)
- **Serverless Compute**: Cloud Run, Cloud Functions
- **Event Bus**: Cloud Pub/Sub
- **Data Storage**: BigQuery, Cloud Storage
- **Security**: Secret Manager, IAM
- **Scheduling**: Cloud Scheduler
- **Monitoring**: Cloud Monitoring, Cloud Logging
- **Frontend**: Retool (custom dashboards)

## Data Flow

1. **Analysis Initiation**:
   - User triggers analysis via Retool dashboard
   - Webhook call is sent to Cloud Run API
   - Analysis request is validated and published to Pub/Sub

2. **Metadata Extraction**:
   - Cloud Function queries INFORMATION_SCHEMA tables
   - Extracts table metadata, schema, usage patterns
   - Stores metadata in Cloud Storage for processing

3. **Optimization Analysis**:
   - Multiple Cloud Functions perform specialized analysis in parallel
   - Results are aggregated in Cloud Storage

4. **Recommendation Generation**:
   - Recommendation engine processes analysis results
   - Generates prioritized recommendations with ROI
   - Stores recommendations in BigQuery

5. **Status Updates**:
   - Cloud Run API provides status updates to Retool
   - Completion notifications are sent when analysis finishes

## Security Considerations

- All services use service accounts with least privilege
- Secret Manager stores sensitive credentials
- All data transfers are encrypted in transit
- API endpoints require authentication
- BigQuery access is controlled via IAM

## Scaling Considerations

- Pub/Sub enables parallel processing of multiple analyses
- Cloud Functions scale automatically based on load
- BigQuery handles datasets of any size efficiently
- Architecture supports analyzing multiple projects simultaneously