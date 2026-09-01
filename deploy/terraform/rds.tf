resource "aws_security_group" "rds" {
  name        = "${var.name}-rds"
  description = "PostgreSQL access from EKS worker nodes only"
  vpc_id      = module.vpc.vpc_id

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.common_tags
}

resource "aws_security_group_rule" "rds_from_eks_nodes" {
  type                     = "ingress"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = module.eks.node_security_group_id
  protocol                 = "tcp"
  from_port                = var.db_port
  to_port                  = var.db_port
  description              = "EKS worker nodes to PostgreSQL"
}

resource "aws_db_subnet_group" "postgres" {
  name       = "${var.name}-postgres"
  subnet_ids = module.vpc.private_subnets
  tags       = local.common_tags
}

resource "aws_db_instance" "postgres" {
  identifier              = "${var.name}-postgres"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 100
  max_allocated_storage   = 500
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = var.db_name
  username                = var.db_username
  manage_master_user_password = true
  port                    = var.db_port
  db_subnet_group_name    = aws_db_subnet_group.postgres.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  publicly_accessible     = false
  multi_az                = true
  backup_retention_period  = 14
  backup_window            = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  copy_tags_to_snapshot   = true
  deletion_protection     = true
  skip_final_snapshot     = var.rds_skip_final_snapshot
  final_snapshot_identifier = var.rds_skip_final_snapshot ? null : "${var.name}-final-${random_id.final_snapshot.hex}"
  auto_minor_version_upgrade = true
  apply_immediately       = false
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  lifecycle {
    prevent_destroy = !var.rds_skip_final_snapshot
  }
}
