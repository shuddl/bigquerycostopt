# Secret Manager module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "secrets" {
  description = "Secrets to create"
  type = list(object({
    name                  = string
    description           = optional(string, "")
    secret_data           = optional(string, null)
    secret_file           = optional(string, null)
    automatic_replication = optional(bool, true)
    regions               = optional(list(string), [])
    labels                = optional(map(string), {})
  }))
  default = []
}

variable "secret_accessors" {
  description = "Members to grant Secret Manager Secret Accessor role"
  type = list(object({
    secret_name = string
    members     = list(string)
  }))
  default = []
}

# Create secrets
resource "google_secret_manager_secret" "secrets" {
  for_each = { for secret in var.secrets : secret.name => secret }

  project   = var.project_id
  secret_id = each.value.name
  
  replication {
    dynamic "user_managed" {
      for_each = each.value.automatic_replication ? [] : [1]
      content {
        dynamic "replicas" {
          for_each = each.value.regions
          content {
            location = replicas.value
          }
        }
      }
    }
    
    dynamic "automatic" {
      for_each = each.value.automatic_replication ? [1] : []
      content {}
    }
  }

  labels = each.value.labels
}

# Set secret values
resource "google_secret_manager_secret_version" "versions" {
  for_each = { for secret in var.secrets : secret.name => secret if secret.secret_data != null || secret.secret_file != null }

  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value.secret_data != null ? each.value.secret_data : file(each.value.secret_file)
}

# Grant Secret Accessor role to specified members
resource "google_secret_manager_secret_iam_binding" "secret_accessors" {
  for_each = { for binding in var.secret_accessors : binding.secret_name => binding }

  project   = var.project_id
  secret_id = google_secret_manager_secret.secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  members   = each.value.members

  depends_on = [google_secret_manager_secret.secrets]
}

output "secret_ids" {
  value = { for k, v in google_secret_manager_secret.secrets : k => v.id }
  description = "The IDs of the created secrets"
}

output "secret_versions" {
  value = { for k, v in google_secret_manager_secret_version.versions : k => v.name }
  description = "The versions of the created secrets"
  sensitive = true
}