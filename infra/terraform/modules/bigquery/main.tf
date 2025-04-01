# BigQuery module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
}

variable "description" {
  description = "Dataset description"
  type        = string
  default     = "BigQuery Cost Intelligence Engine dataset"
}

variable "delete_contents_on_destroy" {
  description = "If true, delete all the tables in the dataset when destroying the resource"
  type        = bool
  default     = false
}

variable "default_table_expiration_ms" {
  description = "Default expiration time for tables in milliseconds"
  type        = number
  default     = null
}

variable "labels" {
  description = "Labels to apply to the dataset"
  type        = map(string)
  default     = {}
}

variable "tables" {
  description = "Tables to create in the dataset"
  type = list(object({
    table_id               = string
    description            = optional(string, "")
    schema_file            = string
    clustering_fields      = optional(list(string), [])
    expiration_time        = optional(number, null)
    deletion_protection    = optional(bool, true)
    partitioning_type      = optional(string, null) # "DAY", "HOUR", "MONTH", or "YEAR"
    partitioning_field     = optional(string, null)
    require_partition_filter = optional(bool, false)
  }))
  default = []
}

variable "views" {
  description = "Views to create in the dataset"
  type = list(object({
    view_id      = string
    description  = optional(string, "")
    query        = string
    labels       = optional(map(string), {})
  }))
  default = []
}

variable "access_roles" {
  description = "Access roles to grant on the dataset"
  type = list(object({
    role          = string
    user_by_email = optional(string, null)
    group_by_email = optional(string, null)
    service_account_id = optional(string, null)
  }))
  default = []
}

# Create BigQuery dataset
resource "google_bigquery_dataset" "dataset" {
  dataset_id                 = var.dataset_id
  project                    = var.project_id
  location                   = var.location
  description                = var.description
  default_table_expiration_ms = var.default_table_expiration_ms
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = var.labels

  # Grant access to specified principals
  dynamic "access" {
    for_each = var.access_roles
    content {
      role = access.value.role
      user_by_email = access.value.user_by_email
      group_by_email = access.value.group_by_email
      service_account_id = access.value.service_account_id
    }
  }
}

# Load table schemas from files
locals {
  schemas = { for table in var.tables : table.table_id => jsondecode(file(table.schema_file)) }
}

# Create BigQuery tables
resource "google_bigquery_table" "tables" {
  for_each = { for table in var.tables : table.table_id => table }

  project             = var.project_id
  dataset_id          = google_bigquery_dataset.dataset.dataset_id
  table_id            = each.value.table_id
  description         = each.value.description
  schema              = jsonencode(local.schemas[each.value.table_id])
  deletion_protection = each.value.deletion_protection
  expiration_time     = each.value.expiration_time

  # Configure clustering if specified
  dynamic "clustering" {
    for_each = length(each.value.clustering_fields) > 0 ? [1] : []
    content {
      fields = each.value.clustering_fields
    }
  }

  # Configure time partitioning if specified
  dynamic "time_partitioning" {
    for_each = each.value.partitioning_type != null ? [1] : []
    content {
      type                     = each.value.partitioning_type
      field                    = each.value.partitioning_field
      require_partition_filter = each.value.require_partition_filter
    }
  }

  depends_on = [google_bigquery_dataset.dataset]
}

# Create BigQuery views
resource "google_bigquery_table" "views" {
  for_each = { for view in var.views : view.view_id => view }

  project     = var.project_id
  dataset_id  = google_bigquery_dataset.dataset.dataset_id
  table_id    = each.value.view_id
  description = each.value.description
  labels      = each.value.labels

  view {
    query          = each.value.query
    use_legacy_sql = false
  }

  deletion_protection = true

  depends_on = [google_bigquery_dataset.dataset, google_bigquery_table.tables]
}

output "dataset_id" {
  value = google_bigquery_dataset.dataset.id
}

output "table_ids" {
  value = { for k, v in google_bigquery_table.tables : k => v.id }
}

output "view_ids" {
  value = { for k, v in google_bigquery_table.views : k => v.id }
}