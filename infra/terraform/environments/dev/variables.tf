# Variables for the development environment

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resource deployment"
  type        = string
  default     = "us-central1"
}

variable "api_version" {
  description = "The version of the API to deploy"
  type        = string
  default     = "latest"
}

variable "api_key" {
  description = "API key for authentication"
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address for alerting"
  type        = string
}

variable "api_server_type" {
  description = "Type of API server to use (flask or fastapi)"
  type        = string
  default     = "fastapi"
}

variable "enable_cost_dashboard" {
  description = "Whether to enable the cost attribution dashboard"
  type        = bool
  default     = true
}