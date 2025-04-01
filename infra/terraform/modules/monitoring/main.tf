# Monitoring module for BigQuery Cost Intelligence Engine

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "notification_channels" {
  description = "Notification channels for alerts"
  type = list(object({
    display_name = string
    type         = string # "email", "slack", "pubsub", etc.
    labels       = map(string)
    description  = optional(string, "")
  }))
  default = []
}

variable "alerts" {
  description = "Alert policies to create"
  type = list(object({
    display_name = string
    combiner     = string # "OR", "AND", "NOT", etc.
    conditions = list(object({
      display_name = string
      condition_threshold = object({
        filter      = string
        duration    = string
        comparison  = string
        threshold_value = number
        aggregations = optional(list(object({
          alignment_period   = optional(string, "60s")
          per_series_aligner = optional(string, "ALIGN_MEAN")
          cross_series_reducer = optional(string, "REDUCE_MEAN")
          group_by_fields    = optional(list(string), [])
        })), [])
      })
    }))
    notification_channels = list(string)
    documentation = optional(object({
      content      = string
      mime_type    = optional(string, "text/markdown")
    }), null)
    enabled     = optional(bool, true)
  }))
  default = []
}

variable "dashboards" {
  description = "Monitoring dashboards to create"
  type = list(object({
    dashboard_name = string
    display_name   = string
    dashboard_json = string  # Path to JSON file with dashboard configuration
  }))
  default = []
}

variable "logs" {
  description = "Log metrics and sinks to create"
  type = object({
    metrics = optional(list(object({
      name        = string
      description = string
      filter      = string
      metric_descriptor = object({
        metric_kind = string  # "GAUGE", "DELTA", "CUMULATIVE"
        value_type  = string  # "BOOL", "INT64", "DOUBLE", "STRING", "DISTRIBUTION"
        unit        = optional(string, "1")
        labels = optional(list(object({
          key         = string
          description = string
        })), [])
      })
    })), [])
    sinks = optional(list(object({
      name            = string
      destination     = string  # e.g., "storage.googleapis.com/my-bucket"
      filter          = string
      include_children = optional(bool, true)
    })), [])
  })
  default = { metrics = [], sinks = [] }
}

variable "uptime_checks" {
  description = "Uptime checks for external HTTP endpoints"
  type = list(object({
    display_name  = string
    resource_type = string  # "uptime_url" or "app_engine_service"
    http_check = object({
      path           = optional(string, "/")
      port           = optional(number, 443)
      use_ssl        = optional(bool, true)
      validate_ssl   = optional(bool, true)
      auth_header = optional(object({
        username     = string
        password     = string
      }), null)
      mask_headers   = optional(bool, false)
      headers        = optional(map(string), {})
    })
    monitored_resource = object({
      type = string  # "uptime_url" or "gae_app"
      labels = map(string)
    })
    period         = optional(string, "60s")
    timeout        = optional(string, "10s")
    content_matchers = optional(list(object({
      content     = string
      matcher     = string  # "CONTAINS", "NOT_CONTAINS", "MATCHES_REGEX", etc.
    })), [])
  }))
  default = []
}

# Create notification channels
resource "google_monitoring_notification_channel" "channels" {
  for_each = { for channel in var.notification_channels : channel.display_name => channel }

  project      = var.project_id
  display_name = each.value.display_name
  type         = each.value.type
  labels       = each.value.labels
  description  = each.value.description
}

# Create alert policies
resource "google_monitoring_alert_policy" "alerts" {
  for_each = { for alert in var.alerts : alert.display_name => alert }

  project      = var.project_id
  display_name = each.value.display_name
  combiner     = each.value.combiner
  enabled      = each.value.enabled

  dynamic "conditions" {
    for_each = each.value.conditions
    content {
      display_name = conditions.value.display_name
      
      condition_threshold {
        filter      = conditions.value.condition_threshold.filter
        duration    = conditions.value.condition_threshold.duration
        comparison  = conditions.value.condition_threshold.comparison
        threshold_value = conditions.value.condition_threshold.threshold_value
        
        dynamic "aggregations" {
          for_each = conditions.value.condition_threshold.aggregations
          content {
            alignment_period   = aggregations.value.alignment_period
            per_series_aligner = aggregations.value.per_series_aligner
            cross_series_reducer = aggregations.value.cross_series_reducer
            group_by_fields    = aggregations.value.group_by_fields
          }
        }
      }
    }
  }

  notification_channels = [
    for channel_name in each.value.notification_channels :
    google_monitoring_notification_channel.channels[channel_name].id
  ]

  dynamic "documentation" {
    for_each = each.value.documentation != null ? [each.value.documentation] : []
    content {
      content   = documentation.value.content
      mime_type = documentation.value.mime_type
    }
  }

  depends_on = [google_monitoring_notification_channel.channels]
}

# Create dashboards
resource "google_monitoring_dashboard" "dashboards" {
  for_each = { for dashboard in var.dashboards : dashboard.dashboard_name => dashboard }

  project          = var.project_id
  dashboard_json   = file(each.value.dashboard_json)
}

# Create log metrics
resource "google_logging_metric" "metrics" {
  for_each = { for metric in var.logs.metrics : metric.name => metric }

  project     = var.project_id
  name        = each.value.name
  description = each.value.description
  filter      = each.value.filter

  metric_descriptor {
    metric_kind = each.value.metric_descriptor.metric_kind
    value_type  = each.value.metric_descriptor.value_type
    unit        = each.value.metric_descriptor.unit

    dynamic "labels" {
      for_each = each.value.metric_descriptor.labels
      content {
        key         = labels.value.key
        description = labels.value.description
      }
    }
  }
}

# Create log sinks
resource "google_logging_project_sink" "sinks" {
  for_each = { for sink in var.logs.sinks : sink.name => sink }

  project      = var.project_id
  name         = each.value.name
  destination  = each.value.destination
  filter       = each.value.filter
  
  unique_writer_identity = true
  include_children       = each.value.include_children
}

# Create uptime checks
resource "google_monitoring_uptime_check_config" "uptime_checks" {
  for_each = { for check in var.uptime_checks : check.display_name => check }

  project      = var.project_id
  display_name = each.value.display_name
  timeout      = each.value.timeout
  period       = each.value.period

  http_check {
    path           = each.value.http_check.path
    port           = each.value.http_check.port
    use_ssl        = each.value.http_check.use_ssl
    validate_ssl   = each.value.http_check.validate_ssl
    mask_headers   = each.value.http_check.mask_headers
    headers        = each.value.http_check.headers
    
    dynamic "auth_info" {
      for_each = each.value.http_check.auth_header != null ? [each.value.http_check.auth_header] : []
      content {
        username = auth_info.value.username
        password = auth_info.value.password
      }
    }
  }

  monitored_resource {
    type   = each.value.monitored_resource.type
    labels = each.value.monitored_resource.labels
  }

  dynamic "content_matchers" {
    for_each = each.value.content_matchers
    content {
      content = content_matchers.value.content
      matcher = content_matchers.value.matcher
    }
  }
}

output "notification_channel_ids" {
  value = { for k, v in google_monitoring_notification_channel.channels : k => v.id }
}

output "alert_policy_ids" {
  value = { for k, v in google_monitoring_alert_policy.alerts : k => v.id }
}

output "dashboard_ids" {
  value = { for k, v in google_monitoring_dashboard.dashboards : k => v.id }
}

output "log_metric_ids" {
  value = { for k, v in google_logging_metric.metrics : k => v.id }
}

output "log_sink_ids" {
  value = { for k, v in google_logging_project_sink.sinks : k => v.id }
}

output "uptime_check_ids" {
  value = { for k, v in google_monitoring_uptime_check_config.uptime_checks : k => v.id }
}