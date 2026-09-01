terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Bootstrap this bucket/table once, then initialize with:
  # terraform init -backend-config="bucket=<state-bucket>" \
  #   -backend-config="key=semantic-router/prod/terraform.tfstate" \
  #   -backend-config="region=<region>"
  backend "s3" {
    encrypt        = true
    use_lockfile   = true
    bucket         = "REPLACE_WITH_TF_STATE_BUCKET"
    key            = "semantic-router/terraform.tfstate"
    region         = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = merge({ Project = var.name, Environment = var.environment, ManagedBy = "terraform" }, var.tags)
  }
}

resource "random_id" "final_snapshot" {
  byte_length = 4
}

locals {
  common_tags = merge({ Project = var.name, Environment = var.environment }, var.tags)
}
