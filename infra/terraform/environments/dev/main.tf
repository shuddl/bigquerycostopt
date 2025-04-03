# Development environment configuration for BigQuery Cost Intelligence Engine

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
  }
  
  backend "gcs" {
    bucket = "bqcostopt-terraform-state-dev"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Local variables
locals {
  environment = "dev"
  
  # Common labels to apply to all resources
  common_labels = {
    environment = local.environment
    managed_by  = "terraform"
    application = "bqcostopt"
  }
  
  # Service account names
  service_accounts = {
    api_service        = "bqcostopt-api-sa"
    worker_service     = "bqcostopt-worker-sa"
    scheduler_service  = "bqcostopt-scheduler-sa"
  }
}

# Create GCS bucket for function source code
resource "google_storage_bucket" "function_source" {
  name     = "${var.project_id}-function-source"
  location = var.region
  labels   = local.common_labels
  
  versioning {
    enabled = true
  }
  
  uniform_bucket_level_access = true
}

# Create GCS bucket for analysis results
resource "google_storage_bucket" "analysis_results" {
  name     = "${var.project_id}-analysis-results"
  location = var.region
  labels   = local.common_labels
  
  lifecycle_rule {
    condition {
      age = 90  # days
    }
    action {
      type = "Delete"
    }
  }
  
  uniform_bucket_level_access = true
}

# Create service accounts with appropriate roles
module "iam" {
  source     = "../../modules/iam"
  project_id = var.project_id
  
  service_accounts = [
    {
      name         = local.service_accounts.api_service
      display_name = "BigQuery Cost Opt API Service Account"
      description  = "Service account for the BigQuery Cost Intelligence Engine API"
      roles        = [
        "roles/pubsub.publisher",
        "roles/bigquery.dataViewer",
        "roles/bigquery.jobUser",
        "roles/monitoring.metricWriter",
        "roles/logging.logWriter"
      ]
    },
    {
      name         = local.service_accounts.worker_service
      display_name = "BigQuery Cost Opt Worker Service Account"
      description  = "Service account for the BigQuery Cost Intelligence Engine worker services"
      roles        = [
        "roles/bigquery.dataViewer",
        "roles/bigquery.jobUser",
        "roles/bigquery.metadataViewer",
        "roles/bigquery.user", 
        "roles/pubsub.subscriber",
        "roles/pubsub.publisher",
        "roles/storage.objectAdmin",
        "roles/monitoring.metricWriter",
        "roles/logging.logWriter"
      ]
    },
    {
      name         = local.service_accounts.scheduler_service
      display_name = "BigQuery Cost Opt Scheduler Service Account"
      description  = "Service account for the BigQuery Cost Intelligence Engine scheduled jobs"
      roles        = [
        "roles/pubsub.publisher",
        "roles/cloudscheduler.serviceAgent"
      ]
    }
  ]
  
  custom_roles = [
    {
      role_id     = "bqCostOptReader"
      title       = "BigQuery Cost Intelligence Engine Reader"
      description = "Custom role for read-only access to BigQuery Cost Intelligence Engine resources"
      permissions = [
        "bigquery.datasets.get",
        "bigquery.tables.get",
        "bigquery.tables.list",
        "storage.objects.get",
        "storage.objects.list"
      ]
    }
  ]
}

# Create Pub/Sub topics and subscriptions
module "pubsub" {
  source     = "../../modules/pubsub"
  project_id = var.project_id
  
  # Analysis request topic
  topic_name = "bqcostopt-analysis-requests"
  labels     = local.common_labels
  subscriptions = [
    {
      name                  = "bqcostopt-analysis-worker-sub"
      ack_deadline_seconds  = 60
      push_endpoint         = "https://${var.region}-${var.project_id}.cloudfunctions.net/bqcostopt-analysis-worker"
      service_account_email = module.iam.service_account_emails[local.service_accounts.worker_service]
    }
  ]
}

# Create CloudRun service for API
module "api_service" {
  source     = "../../modules/cloud_run"
  project_id = var.project_id
  region     = var.region
  
  service_name          = "bqcostopt-api"
  image_name            = "gcr.io/${var.project_id}/bqcostopt-api"
  image_tag             = var.api_version
  service_account_email = module.iam.service_account_emails[local.service_accounts.api_service]
  
  env_vars = {
    GCP_PROJECT_ID           = var.project_id
    ANALYSIS_REQUEST_TOPIC   = module.pubsub.topic_name
    ENVIRONMENT              = local.environment
    API_SERVER_TYPE          = var.api_server_type
    WEBHOOK_SECRET           = "secret-${local.environment}"
    API_KEY                  = var.api_key
    CACHE_TTL_SECONDS        = "300"
    ML_ENABLED               = "true"
    ENABLE_COST_DASHBOARD    = var.enable_cost_dashboard
  }
  
  secret_env_vars = []
  
  cpu             = "1"
  memory          = "2Gi"
  concurrency     = 80
  timeout         = 300
  max_instances   = 10
  min_instances   = 1
}

# Create BigQuery dataset for storing results
module "bigquery" {
  source     = "../../modules/bigquery"
  project_id = var.project_id
  location   = "US"
  
  dataset_id                  = "bqcostopt"
  description                 = "BigQuery Cost Intelligence Engine dataset"
  delete_contents_on_destroy  = false
  labels                      = local.common_labels
  
  tables = [
    {
      table_id         = "analysis_results"
      description      = "Results of BigQuery cost optimization analyses"
      schema_file      = "${path.module}/schemas/analysis_results.json"
      clustering_fields = ["project_id", "dataset_id"]
      partitioning_type = "DAY"
      partitioning_field = "analysis_date"
    },
    {
      table_id         = "recommendations"
      description      = "Cost optimization recommendations"
      schema_file      = "${path.module}/schemas/recommendations.json"
      clustering_fields = ["recommendation_type", "project_id"]
      partitioning_type = "DAY"
      partitioning_field = "creation_date"
    },
    {
      table_id         = "implementation_history"
      description      = "History of implemented recommendations"
      schema_file      = "${path.module}/schemas/implementation_history.json"
      partitioning_type = "MONTH"
      partitioning_field = "implementation_date"
    },
    {
      table_id         = "cost_attribution"
      description      = "BigQuery cost attribution data by user, team, and query pattern"
      schema_file      = null
      clustering_fields = ["project_id", "user_email"]
      partitioning_type = "DAY"
      partitioning_field = "date"
    },
    {
      table_id         = "cost_anomalies"
      description      = "Detected cost anomalies"
      schema_file      = null
      clustering_fields = ["anomaly_type", "project_id"]
      partitioning_type = "DAY"
      partitioning_field = "detection_date"
    },
    {
      table_id         = "cost_forecasts"
      description      = "Cost forecasts"
      schema_file      = null
      partitioning_type = "DAY"
      partitioning_field = "generation_date"
    },
    {
      table_id         = "team_mapping"
      description      = "Mapping of users to teams for cost attribution"
      schema_file      = null
      clustering_fields = ["team"]
    }
  ]
  
  views = [
    {
      view_id      = "active_recommendations"
      description  = "Active recommendations that haven't been implemented yet"
      query        = file("${path.module}/sql/active_recommendations.sql")
    },
    {
      view_id      = "savings_by_project"
      description  = "Estimated savings by project"
      query        = file("${path.module}/sql/savings_by_project.sql")
    },
    {
      view_id      = "cost_by_team"
      description  = "Cost attribution by team"
      query        = "SELECT date, team, SUM(total_cost_usd) as total_cost_usd, COUNT(DISTINCT user_email) as user_count, SUM(query_count) as query_count FROM `${var.project_id}.bqcostopt.cost_attribution` GROUP BY date, team ORDER BY date DESC, total_cost_usd DESC"
    },
    {
      view_id      = "cost_by_user"
      description  = "Cost attribution by user"
      query        = "SELECT date, user_email, team, SUM(total_cost_usd) as total_cost_usd, SUM(query_count) as query_count FROM `${var.project_id}.bqcostopt.cost_attribution` GROUP BY date, user_email, team ORDER BY date DESC, total_cost_usd DESC"
    },
    {
      view_id      = "cost_trends_daily"
      description  = "Daily cost trends"
      query        = "SELECT date, SUM(total_cost_usd) as total_cost_usd, COUNT(DISTINCT user_email) as user_count, COUNT(DISTINCT team) as team_count, SUM(query_count) as query_count FROM `${var.project_id}.bqcostopt.cost_attribution` GROUP BY date ORDER BY date DESC"
    },
    {
      view_id      = "recent_anomalies"
      description  = "Recent cost anomalies"
      query        = "SELECT * FROM `${var.project_id}.bqcostopt.cost_anomalies` WHERE detection_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) ORDER BY detection_date DESC, severity DESC"
    }
  ]
  
  access_roles = [
    {
      role          = "roles/bigquery.dataViewer"
      service_account_id = module.iam.service_account_emails[local.service_accounts.worker_service]
    }
  ]
}

# Create Cloud Functions for worker processes
module "analysis_worker" {
  source     = "../../modules/cloud_functions"
  project_id = var.project_id
  region     = var.region
  
  function_name         = "bqcostopt-analysis-worker"
  description           = "Worker function for processing BigQuery dataset analysis requests"
  runtime               = "python39"
  entry_point           = "process_analysis_request"
  source_dir            = "../../../function_source/analysis_worker"
  bucket_name           = google_storage_bucket.function_source.name
  service_account_email = module.iam.service_account_emails[local.service_accounts.worker_service]
  
  env_vars = {
    GCP_PROJECT_ID        = var.project_id
    RESULTS_BUCKET        = google_storage_bucket.analysis_results.name
    BIGQUERY_DATASET      = module.bigquery.dataset_id
    ENVIRONMENT           = local.environment
  }
  
  memory         = 1024
  timeout        = 540  # 9 minutes
  max_instances  = 10
  
  event_trigger = {
    event_type   = "google.pubsub.topic.publish"
    resource     = module.pubsub.topic_id
    retry_policy = true
  }
}

# Create monitoring resources
module "monitoring" {
  source     = "../../modules/monitoring"
  project_id = var.project_id
  
  notification_channels = [
    {
      display_name = "Email Alerts"
      type         = "email"
      labels       = {
        email_address = var.alert_email
      }
    }
  ]
  
  alerts = [
    {
      display_name = "API Error Rate High"
      combiner     = "OR"
      conditions = [
        {
          display_name = "Error rate exceeds threshold"
          condition_threshold = {
            filter      = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"bqcostopt-api\" AND metric.type = \"logging.googleapis.com/log_entry_count\" AND metric.labels.severity = \"ERROR\""
            duration    = "300s"
            comparison  = "COMPARISON_GT"
            threshold_value = 5
            aggregations = [
              {
                alignment_period = "300s"
                per_series_aligner = "ALIGN_RATE"
              }
            ]
          }
        }
      ]
      notification_channels = ["Email Alerts"]
      documentation = {
        content = "The BigQuery Cost Intelligence Engine API is experiencing a high error rate. Check the logs for details."
      }
    },
    {
      display_name = "Analysis Worker Failures"
      combiner     = "OR"
      conditions = [
        {
          display_name = "Worker failure rate exceeds threshold"
          condition_threshold = {
            filter      = "resource.type = \"cloud_function\" AND resource.labels.function_name = \"bqcostopt-analysis-worker\" AND metric.type = \"cloudfunctions.googleapis.com/function/execution_count\" AND metric.labels.status = \"error\""
            duration    = "600s"
            comparison  = "COMPARISON_GT"
            threshold_value = 3
            aggregations = [
              {
                alignment_period = "600s"
                per_series_aligner = "ALIGN_RATE"
              }
            ]
          }
        }
      ]
      notification_channels = ["Email Alerts"]
      documentation = {
        content = "The BigQuery Cost Intelligence Engine analysis worker is experiencing failures. Check the function logs for details."
      }
    }
  ]
  
  dashboards = [
    {
      dashboard_name = "bqcostopt-overview"
      display_name   = "BigQuery Cost Intelligence Engine Overview"
      dashboard_json = "${path.module}/dashboards/overview.json"
    }
  ]
  
  uptime_checks = [
    {
      display_name  = "BQ Cost Opt API Health Check"
      resource_type = "uptime_url"
      http_check = {
        path = "/api/v1/health"
        port = 443
        use_ssl = true
        headers = {
          "X-API-Key" = var.api_key
        }
      }
      monitored_resource = {
        type = "uptime_url"
        labels = {
          host = "${module.api_service.url}"
          project_id = var.project_id
        }
      }
      period = "60s"
      timeout = "10s"
      content_matchers = [
        {
          content = "\"status\":\"OK\""
          matcher = "CONTAINS"
        }
      ]
    }
  ]
}

# Create secrets
module "secrets" {
  source     = "../../modules/secrets"
  project_id = var.project_id
  
  secrets = [
    {
      name = "api-key"
      description = "API key for the BigQuery Cost Intelligence Engine"
      secret_data = var.api_key
      labels      = local.common_labels
    }
  ]
  
  secret_accessors = [
    {
      secret_name = "api-key"
      members     = ["serviceAccount:${module.iam.service_account_emails[local.service_accounts.api_service]}"]
    }
  ]
}

# Output important values
output "api_url" {
  value = module.api_service.url
  description = "The URL of the API service"
}

output "analysis_worker_function" {
  value = module.analysis_worker.function_name
  description = "The name of the analysis worker function"
}

output "dataset_id" {
  value = module.bigquery.dataset_id
  description = "The BigQuery dataset ID"
}

output "service_account_emails" {
  value = module.iam.service_account_emails
  description = "Service account emails"
}