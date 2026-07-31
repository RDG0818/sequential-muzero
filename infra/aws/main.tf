# AWS IaC sketch for sequential-muzero — DEMONSTRATION ONLY, never applied.
#
# Provisions a single GPU instance running the whole Ray cluster (learner +
# data actors + reanalyze actor + replay buffer), matching how this repo
# actually runs today: every actor touches the one visible GPU via JAX
# (confirmed in profiling logs — DataActor/ReanalyzeActor aren't CPU-only
# despite their Ray `num_cpus`-only resource request), so a naive
# GPU-node/CPU-node split would silently break MCTS inference on the CPU
# node. Splitting the cluster across nodes is a real future extension, but
# needs a code change first (CPU JAX backend fallback for DataActor) — see
# docs/deployment.md.
#
# Terraform's own Ray autoscaler integration is also skipped here to keep
# this lean; noted as a future extension in docs/deployment.md, not built.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type. g5.2xlarge = 1x A10G (24GB VRAM), 8 vCPU — comfortably covers this repo's 3060 Ti/5070 Ti dev boxes."
  type        = string
  default     = "g5.2xlarge"
}

variable "key_name" {
  description = "Existing EC2 key pair name for SSH access"
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in and hit the Ray dashboard. Restrict to your own IP/32 before ever applying this for real."
  type        = string
  default     = "0.0.0.0/0"
}

variable "git_ref" {
  description = "Branch/tag/commit to check out on boot"
  type        = string
  default     = "updates"
}

variable "wandb_api_key" {
  description = "wandb API key for online logging; leave empty to run with wandb disabled"
  type        = string
  default     = ""
  sensitive   = true
}

variable "root_volume_gb" {
  description = "Root EBS volume size — replay buffer + checkpoints live here"
  type        = number
  default     = 100
}

# AWS Deep Learning AMI: NVIDIA driver, Docker, and the NVIDIA Container
# Toolkit all preinstalled — avoids hand-rolling CUDA driver install in
# user_data, which is the single easiest thing to get wrong in a GPU IaC
# sketch.
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
}

resource "aws_security_group" "muzero" {
  name        = "sequential-muzero-sg"
  description = "SSH + Ray dashboard for the sequential-muzero training instance"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  ingress {
    description = "Ray dashboard"
    from_port   = 8265
    to_port     = 8265
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "muzero_trainer" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.muzero.id]

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/bootstrap.sh.tpl", {
    git_repo_url  = "https://github.com/RDG0818/sequential-muzero.git"
    git_ref       = var.git_ref
    wandb_api_key = var.wandb_api_key
  })

  tags = {
    Name = "sequential-muzero-trainer"
  }
}

output "instance_public_ip" {
  value = aws_instance.muzero_trainer.public_ip
}

output "ray_dashboard_url" {
  value = "http://${aws_instance.muzero_trainer.public_ip}:8265"
}
