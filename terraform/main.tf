# AWS Infrastructure for Astrobiology AI Training
# ================================================
# Terraform configuration for optimized training infrastructure

terraform {
  required_version = ">= 1.0"
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

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "astrobiology-ai"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "key_pair_name" {
  description = "EC2 Key Pair name"
  type        = string
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Random ID for unique resource names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name        = "${var.project_name}-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = 2
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "${var.project_name}-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = 2
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name        = "${var.project_name}-private-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"
  
  tags = {
    Name        = "${var.project_name}-nat-eip"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  
  tags = {
    Name        = "${var.project_name}-nat-gateway"
    Environment = var.environment
  }
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-private-rt"
    Environment = var.environment
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# Security Groups
resource "aws_security_group" "training_sg" {
  name_prefix = "${var.project_name}-training-"
  vpc_id      = aws_vpc.main.id
  
  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Jupyter notebook
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  
  # TensorBoard
  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  
  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-training-sg"
    Environment = var.environment
  }
}

# S3 Buckets
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-data-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-data-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-models-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-model-bucket"
    Environment = var.environment
  }
}

# IAM Role for EC2 instances
resource "aws_iam_role" "training_role" {
  name = "${var.project_name}-training-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "training_policy" {
  name = "${var.project_name}-training-policy"
  role = aws_iam_role.training_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*",
          aws_s3_bucket.model_bucket.arn,
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training_profile" {
  name = "${var.project_name}-training-profile"
  role = aws_iam_role.training_role.name
}

# Launch Template for Training Instances
resource "aws_launch_template" "training_template" {
  name_prefix   = "${var.project_name}-training-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = "p3.2xlarge"  # Single GPU for development
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.training_sg.id]
  
  iam_instance_profile {
    name = aws_iam_instance_profile.training_profile.name
  }
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_data_bucket  = aws_s3_bucket.data_bucket.bucket
    s3_model_bucket = aws_s3_bucket.model_bucket.bucket
    aws_region      = var.aws_region
  }))
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.project_name}-training-instance"
      Environment = var.environment
    }
  }
}

# Auto Scaling Group for Training Instances
resource "aws_autoscaling_group" "training_asg" {
  name                = "${var.project_name}-training-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = []
  health_check_type   = "EC2"
  
  min_size         = 0
  max_size         = 3
  desired_capacity = 1
  
  launch_template {
    id      = aws_launch_template.training_template.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-training-asg"
    propagate_at_launch = false
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "training_logs" {
  name              = "/aws/ec2/${var.project_name}-training"
  retention_in_days = 14
  
  tags = {
    Name        = "${var.project_name}-training-logs"
    Environment = var.environment
  }
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "data_bucket_name" {
  description = "S3 data bucket name"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "model_bucket_name" {
  description = "S3 model bucket name"
  value       = aws_s3_bucket.model_bucket.bucket
}

output "training_security_group_id" {
  description = "Training security group ID"
  value       = aws_security_group.training_sg.id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}
