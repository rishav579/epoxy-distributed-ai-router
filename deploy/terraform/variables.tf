variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Short, globally unique prefix for resource names."
  type        = string
  default     = "semantic-router"
  validation {
    condition     = can(regex("^[a-z0-9-]{3,24}$", var.name))
    error_message = "name must be 3-24 lowercase letters, numbers, or hyphens."
  }
}

variable "environment" {
  type        = string
  description = "Deployment environment used for tagging and naming."
  default     = "prod"
}

variable "vpc_cidr" {
  type        = string
  description = "Primary VPC CIDR."
  default     = "10.40.0.0/16"
}

variable "availability_zones" {
  type        = list(string)
  description = "At least two AZs for highly available private subnets."
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "Provide at least two availability zones."
  }
}

variable "eks_kubernetes_version" {
  type        = string
  description = "EKS Kubernetes version. Pin and upgrade deliberately."
  default     = "1.31"
}

variable "worker_instance_types" {
  type        = list(string)
  description = "EC2 instance types for inference worker nodes."
  default     = ["m6i.large"]
}

variable "worker_min_size" {
  type    = number
  default = 1
}

variable "worker_desired_size" {
  type    = number
  default = 2
}

variable "worker_max_size" {
  type    = number
  default = 20
}

variable "db_instance_class" {
  type        = string
  description = "RDS instance class."
  default     = "db.t4g.small"
}

variable "db_name" {
  type    = string
  default = "inference"
}

variable "db_username" {
  type      = string
  sensitive = true
  default   = "inference"
}

variable "db_port" {
  type    = number
  default = 5432
}

variable "rds_skip_final_snapshot" {
  type        = bool
  description = "Only set true for ephemeral environments."
  default     = false
}

variable "worker_namespace" {
  type    = string
  default = "inference"
}

variable "worker_service_account" {
  type    = string
  default = "inference-worker"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags applied to all taggable resources."
  default     = {}
}

variable "github_actions_repo" {
  type        = string
  description = "GitHub repository authorized for OIDC deployment in owner/repo format."
  default     = "rishav579/epoxy-distributed-ai-router"
}

variable "create_github_oidc_provider" {
  type        = bool
  description = "Whether to create the GitHub Actions OIDC provider (set to false if already present in AWS account)."
  default     = true
}
