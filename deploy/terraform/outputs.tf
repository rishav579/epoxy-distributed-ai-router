output "vpc_id" {
  value = module.vpc.vpc_id
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value     = module.eks.cluster_endpoint
  sensitive = true
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.address
}

output "rds_master_secret_arn" {
  value     = aws_db_instance.postgres.master_user_secret[0].secret_arn
  sensitive = true
}

output "model_artifacts_bucket" {
  value = aws_s3_bucket.ml_artifacts.bucket
}

output "worker_irsa_role_arn" {
  value = aws_iam_role.worker.arn
}

output "worker_service_account_annotation" {
  value = {
    "eks.amazonaws.com/role-arn" = aws_iam_role.worker.arn
  }
}

output "worker_s3_prefixes" {
  value = {
    lora   = "s3://${aws_s3_bucket.ml_artifacts.bucket}/lora/"
    mlflow = "s3://${aws_s3_bucket.ml_artifacts.bucket}/mlflow/"
  }
}

output "github_actions_role_arn" {
  description = "ARN of the IAM role to configure in GitHub repository secrets as AWS_GITHUB_ACTIONS_ROLE_ARN."
  value       = aws_iam_role.github_actions.arn
}
