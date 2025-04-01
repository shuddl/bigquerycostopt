# Cloud Functions module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

variable "function_name" {
  description = "Cloud Function name"
  type        = string
}

variable "description" {
  description = "Function description"
  type        = string
  default     = ""
}

variable "runtime" {
  description = "Runtime environment"
  type        = string
  default     = "python39"
}

variable "entry_point" {
  description = "Function entry point"
  type        = string
}

variable "source_dir" {
  description = "Source directory"
  type        = string
}

variable "bucket_name" {
  description = "GCS bucket to store function source"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for the function"
  type        = string
}

variable "env_vars" {
  description = "Environment variables"
  type        = map(string)
  default     = {}
}

variable "secret_env_vars" {
  description = "Secret environment variables from Secret Manager"
  type = list(object({
    key     = string
    secret  = string
    version = string
  }))
  default = []
}

variable "memory" {
  description = "Memory allocation in MB"
  type        = number
  default     = 256
}

variable "timeout" {
  description = "Function timeout in seconds"
  type        = number
  default     = 60
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "vpc_connector" {
  description = "VPC connector name"
  type        = string
  default     = null
}

variable "vpc_egress" {
  description = "VPC egress setting"
  type        = string
  default     = "PRIVATE_RANGES_ONLY"
}

variable "event_trigger" {
  description = "Event trigger configuration"
  type = object({
    event_type   = string
    resource     = string
    retry_policy = optional(bool, false)
  })
  default = null
}

# Create zip archive of source code
data "archive_file" "source" {
  type        = "zip"
  source_dir  = var.source_dir
  output_path = "/tmp/${var.function_name}-source.zip"
}

# Upload source code to GCS bucket
resource "google_storage_bucket_object" "source" {
  name   = "${var.function_name}/${data.archive_file.source.output_md5}.zip"
  bucket = var.bucket_name
  source = data.archive_file.source.output_path
}

# Create Cloud Function
resource "google_cloudfunctions_function" "function" {
  name                  = var.function_name
  description           = var.description
  runtime               = var.runtime
  project               = var.project_id
  region                = var.region
  available_memory_mb   = var.memory
  source_archive_bucket = var.bucket_name
  source_archive_object = google_storage_bucket_object.source.name
  entry_point           = var.entry_point
  service_account_email = var.service_account_email
  timeout               = var.timeout
  max_instances         = var.max_instances
  min_instances         = var.min_instances

  environment_variables = var.env_vars

  # Add secret environment variables
  dynamic "secret_environment_variables" {
    for_each = var.secret_env_vars
    content {
      key     = secret_environment_variables.value.key
      secret  = secret_environment_variables.value.secret
      version = secret_environment_variables.value.version
    }
  }

  # Configure VPC connector if provided
  dynamic "vpc_connector" {
    for_each = var.vpc_connector != null ? [1] : []
    content {
      name = var.vpc_connector
      egress_settings = var.vpc_egress
    }
  }

  # Configure event trigger if provided
  dynamic "event_trigger" {
    for_each = var.event_trigger != null ? [1] : []
    content {
      event_type     = var.event_trigger.event_type
      resource       = var.event_trigger.resource
      failure_policy {
        retry = var.event_trigger.retry_policy
      }
    }
  }
}

output "function_id" {
  value = google_cloudfunctions_function.function.id
}

output "function_name" {
  value = google_cloudfunctions_function.function.name
}

output "function_url" {
  value = google_cloudfunctions_function.function.https_trigger_url
}