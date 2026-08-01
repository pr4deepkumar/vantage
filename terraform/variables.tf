variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "app_name" {
  description = "Application name identifier"
  type        = string
  default     = "vantage"
}

variable "db_password" {
  description = "Password for PostgreSQL RDS database instance"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key for live evaluation calls"
  type        = string
  sensitive   = true
  default     = ""
}

variable "container_cpu" {
  description = "CPU units for ECS task (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "container_memory" {
  description = "Memory for ECS task in MB"
  type        = number
  default     = 1024
}
