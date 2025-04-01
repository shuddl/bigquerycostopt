# Production environment configuration for BigQuery Cost Intelligence Engine

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
    bucket = "bqcostopt-terraform-state-prod"
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
  environment = "prod"
  
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
      age = 365  # days
    }
    action {
      type = "Delete"
    }
  }
  
  uniform_bucket_level_access = true
}

# Create VPC connector for private networking
resource "google_vpc_access_connector" "connector" {
  name          = "bqcostopt-vpc-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = "default"
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
      ack_deadline_seconds  = 120
      push_endpoint         = "https://${var.region}-${var.project_id}.cloudfunctions.net/bqcostopt-analysis-worker"
      service_account_email = module.iam.service_account_emails[local.service_accounts.worker_service]
    }
  ]
}

# Create CloudRun service for API with autoscaling and high availability
module "api_service" {
  source     = "../../modules/cloud_run"
  project_id = var.project_id
  region     = var.region
  
  service_name          = "bqcostopt-api"
  image_name            = "gcr.io/${var.project_id}/bqcostopt-api"
  image_tag             = var.api_version
  service_account_email = module.iam.service_account_emails[local.service_accounts.api_service]
  
  env_vars = {
    GCP_PROJECT_ID        = var.project_id
    ANALYSIS_REQUEST_TOPIC = module.pubsub.topic_name
    ENVIRONMENT           = local.environment
  }
  
  secret_env_vars = [
    {
      name    = "API_KEY"
      secret  = "api-key"
      version = "latest"
    }
  ]
  
  cpu             = "2"
  memory          = "2Gi"
  concurrency     = 80
  timeout         = 300
  max_instances   = 25
  min_instances   = 3
  
  vpc_connector   = google_vpc_access_connector.connector.id
  vpc_egress      = "private-ranges-only"
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
      schema_file      = "${path.module}/../dev/schemas/analysis_results.json"
      clustering_fields = ["project_id", "dataset_id"]
      partitioning_type = "DAY"
      partitioning_field = "analysis_date"
    },
    {
      table_id         = "recommendations"
      description      = "Cost optimization recommendations"
      schema_file      = "${path.module}/../dev/schemas/recommendations.json"
      clustering_fields = ["recommendation_type", "project_id"]
      partitioning_type = "DAY"
      partitioning_field = "creation_date"
    },
    {
      table_id         = "implementation_history"
      description      = "History of implemented recommendations"
      schema_file      = "${path.module}/../dev/schemas/implementation_history.json"
      partitioning_type = "MONTH"
      partitioning_field = "implementation_date"
    }
  ]
  
  views = [
    {
      view_id      = "active_recommendations"
      description  = "Active recommendations that haven't been implemented yet"
      query        = file("${path.module}/../dev/sql/active_recommendations.sql")
    },
    {
      view_id      = "savings_by_project"
      description  = "Estimated savings by project"
      query        = file("${path.module}/../dev/sql/savings_by_project.sql")
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
  
  memory         = 2048
  timeout        = 540  # 9 minutes
  max_instances  = 20
  
  vpc_connector  = google_vpc_access_connector.connector.id
  vpc_egress     = "PRIVATE_RANGES_ONLY"
  
  event_trigger = {
    event_type   = "google.pubsub.topic.publish"
    resource     = module.pubsub.topic_id
    retry_policy = true
  }
}

# Create scheduled jobs for regular analysis
resource "google_cloud_scheduler_job" "daily_analysis" {
  name             = "bqcostopt-daily-analysis"
  description      = "Trigger daily analysis of key datasets"
  schedule         = "0 3 * * *"  # Run at 3 AM every day
  time_zone        = "UTC"
  attempt_deadline = "1800s"
  
  pubsub_target {
    topic_name = module.pubsub.topic_id
    data       = base64encode(jsonencode({
      "analysis_id": "scheduled-daily-${formatdate("YYYY-MM-DD", timestamp())}",
      "project_id": var.project_id,
      "dataset_id": var.key_dataset_id,
      "scheduled": true
    }))
    
    attributes = {
      "origin": "scheduler"
    }
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
    },
    {
      display_name = "PagerDuty Critical"
      type         = "pagerduty"
      labels       = {
        service_key = var.pagerduty_service_key
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
      notification_channels = ["Email Alerts", "PagerDuty Critical"]
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
    },
    {
      display_name = "API Latency High"
      combiner     = "OR"
      conditions = [
        {
          display_name = "Latency exceeds threshold"
          condition_threshold = {
            filter      = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"bqcostopt-api\" AND metric.type = \"run.googleapis.com/request_latencies\""
            duration    = "300s"
            comparison  = "COMPARISON_GT"
            threshold_value = 2000
            aggregations = [
              {
                alignment_period = "300s"
                per_series_aligner = "ALIGN_PERCENTILE_95"
              }
            ]
          }
        }
      ]
      notification_channels = ["Email Alerts"]
      documentation = {
        content = "The BigQuery Cost Intelligence Engine API is experiencing high latency. Check for potential performance issues."
      }
    }
  ]
  
  dashboards = [
    {
      dashboard_name = "bqcostopt-overview"
      display_name   = "BigQuery Cost Intelligence Engine Overview"
      dashboard_json = "${path.module}/../dev/dashboards/overview.json"
    }
  ]
  
  logs = {
    metrics = [
      {
        name        = "bqcostopt_analysis_count"
        description = "Count of analyses performed"
        filter      = "resource.type=\"cloud_function\" AND resource.labels.function_name=\"bqcostopt-analysis-worker\" AND jsonPayload.event=\"analysis_complete\""
        metric_descriptor = {
          metric_kind = "DELTA"
          value_type  = "INT64"
          labels = [
            {
              key = "status"
              description = "Analysis status"
            },
            {
              key = "project_id"
              description = "Project ID"
            }
          ]
        }
      }
    ],
    sinks = [
      {
        name        = "bqcostopt-errors-to-bigquery"
        destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/bqcostopt_logs"
        filter      = "resource.labels.service_name=\"bqcostopt-api\" OR resource.labels.function_name=\"bqcostopt-analysis-worker\" severity>=ERROR"
      }
    ]
  }
  
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