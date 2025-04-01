# IAM module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "service_accounts" {
  description = "Service accounts to create"
  type = list(object({
    name        = string
    display_name = string
    description = string
    roles       = list(string)
  }))
  default = []
}

variable "custom_roles" {
  description = "Custom IAM roles to create"
  type = list(object({
    role_id      = string
    title        = string
    description  = string
    permissions  = list(string)
  }))
  default = []
}

variable "member_roles" {
  description = "Role bindings for existing members"
  type = list(object({
    member      = string # e.g., "user:jane@example.com", "group:admins@example.com"
    roles       = list(string)
  }))
  default = []
}

# Create service accounts
resource "google_service_account" "service_accounts" {
  for_each = { for sa in var.service_accounts : sa.name => sa }

  account_id   = each.value.name
  display_name = each.value.display_name
  description  = each.value.description
  project      = var.project_id
}

# Create custom IAM roles
resource "google_project_iam_custom_role" "custom_roles" {
  for_each = { for role in var.custom_roles : role.role_id => role }

  role_id     = each.value.role_id
  title       = each.value.title
  description = each.value.description
  permissions = each.value.permissions
  project     = var.project_id
}

# Grant roles to service accounts
resource "google_project_iam_member" "service_account_roles" {
  for_each = {
    for pair in flatten([
      for sa in var.service_accounts : [
        for role in sa.roles : {
          sa_name = sa.name
          role    = role
        }
      ]
    ]) : "${pair.sa_name}-${pair.role}" => pair
  }

  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${google_service_account.service_accounts[each.value.sa_name].email}"

  depends_on = [google_service_account.service_accounts]
}

# Grant roles to existing members
resource "google_project_iam_member" "member_roles" {
  for_each = {
    for pair in flatten([
      for binding in var.member_roles : [
        for role in binding.roles : {
          member = binding.member
          role   = role
        }
      ]
    ]) : "${pair.member}-${pair.role}" => pair
  }

  project = var.project_id
  role    = each.value.role
  member  = each.value.member
}

# Generate service account keys (optional, consider Secret Manager or Workload Identity instead)
resource "google_service_account_key" "keys" {
  for_each = { for sa in var.service_accounts : sa.name => sa if lookup(sa, "create_key", false) }

  service_account_id = google_service_account.service_accounts[each.key].name
  private_key_type   = "TYPE_GOOGLE_CREDENTIALS_FILE"
}

output "service_account_emails" {
  value = { for k, v in google_service_account.service_accounts : k => v.email }
  description = "The emails of the created service accounts"
}

output "service_account_ids" {
  value = { for k, v in google_service_account.service_accounts : k => v.id }
  description = "The IDs of the created service accounts"
}

output "custom_role_ids" {
  value = { for k, v in google_project_iam_custom_role.custom_roles : k => v.id }
  description = "The IDs of the created custom roles"
}