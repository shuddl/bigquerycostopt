# Pub/Sub module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "topic_name" {
  description = "Pub/Sub topic name"
  type        = string
}

variable "subscriptions" {
  description = "Pub/Sub subscriptions to create for the topic"
  type = list(object({
    name                       = string
    ack_deadline_seconds       = optional(number, 20)
    message_retention_duration = optional(string, "604800s") # 7 days
    retain_acked_messages      = optional(bool, false)
    filter                     = optional(string, "")
    push_endpoint              = optional(string, "")
    service_account_email      = optional(string, "")
  }))
  default = []
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}

# Create Pub/Sub topic
resource "google_pubsub_topic" "topic" {
  name    = var.topic_name
  project = var.project_id
  labels  = var.labels

  message_retention_duration = "604800s" # 7 days
}

# Create Pub/Sub subscriptions
resource "google_pubsub_subscription" "subscriptions" {
  for_each = { for s in var.subscriptions : s.name => s }

  name    = each.value.name
  topic   = google_pubsub_topic.topic.name
  project = var.project_id
  labels  = var.labels

  ack_deadline_seconds       = each.value.ack_deadline_seconds
  message_retention_duration = each.value.message_retention_duration
  retain_acked_messages      = each.value.retain_acked_messages

  dynamic "push_config" {
    for_each = each.value.push_endpoint != "" ? [1] : []
    content {
      push_endpoint = each.value.push_endpoint
      
      dynamic "oidc_token" {
        for_each = each.value.service_account_email != "" ? [1] : []
        content {
          service_account_email = each.value.service_account_email
        }
      }
    }
  }

  filter = each.value.filter != "" ? each.value.filter : null

  # Configure exponential backoff for retries
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"  # 10 minutes
  }

  # Configure dead letter policy
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }

  depends_on = [google_pubsub_topic.dead_letter]
}

# Create dead letter topic for failed messages
resource "google_pubsub_topic" "dead_letter" {
  name    = "${var.topic_name}-dead-letter"
  project = var.project_id
  labels  = var.labels
}

# Create a subscription for the dead letter topic for monitoring
resource "google_pubsub_subscription" "dead_letter_subscription" {
  name    = "${var.topic_name}-dead-letter-sub"
  topic   = google_pubsub_topic.dead_letter.name
  project = var.project_id
  labels  = var.labels

  ack_deadline_seconds       = 20
  message_retention_duration = "604800s" # 7 days
  retain_acked_messages      = true
}

output "topic_id" {
  value = google_pubsub_topic.topic.id
}

output "topic_name" {
  value = google_pubsub_topic.topic.name
}

output "subscriptions" {
  value = { for k, v in google_pubsub_subscription.subscriptions : k => v.id }
}

output "dead_letter_topic_id" {
  value = google_pubsub_topic.dead_letter.id
}

output "dead_letter_subscription_id" {
  value = google_pubsub_subscription.dead_letter_subscription.id
}