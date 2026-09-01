data "aws_iam_policy_document" "worker_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub"
      values   = ["system:serviceaccount:${var.worker_namespace}:${var.worker_service_account}"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "${var.name}-worker-s3"
  assume_role_policy = data.aws_iam_policy_document.worker_assume_role.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "worker_s3" {
  statement {
    sid       = "ListModelArtifacts"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.ml_artifacts.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["lora/*", "mlflow/*"]
    }
  }
  statement {
    sid       = "ReadModelArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.ml_artifacts.arn}/lora/*", "${aws_s3_bucket.ml_artifacts.arn}/mlflow/*"]
  }
}

resource "aws_iam_role_policy" "worker_s3" {
  name   = "read-model-artifacts"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker_s3.json
}

resource "aws_s3_bucket" "ml_artifacts" {
  bucket_prefix = "${var.name}-ml-"
  force_destroy = false
  tags          = local.common_tags
}

resource "aws_s3_bucket_public_access_block" "ml_artifacts" {
  bucket                  = aws_s3_bucket.ml_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_policy" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  policy = data.aws_iam_policy_document.ml_artifacts_policy.json
}

data "aws_iam_policy_document" "ml_artifacts_policy" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.ml_artifacts.arn, "${aws_s3_bucket.ml_artifacts.arn}/*"]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}
