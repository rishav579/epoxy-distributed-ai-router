module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name                   = var.name
  cluster_version                = var.eks_kubernetes_version
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  control_plane_subnet_ids       = module.vpc.private_subnets
  cluster_endpoint_public_access = false
  cluster_endpoint_private_access = true

  enable_irsa = true
  enable_cluster_creator_admin_permissions = true

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
  }

  eks_managed_node_groups = {
    inference = {
      name           = "inference"
      instance_types = var.worker_instance_types
      min_size       = var.worker_min_size
      desired_size   = var.worker_desired_size
      max_size       = var.worker_max_size
      subnet_ids     = module.vpc.private_subnets
      capacity_type  = "ON_DEMAND"
      labels         = { workload = "inference" }
    }
  }

  tags = local.common_tags
}
