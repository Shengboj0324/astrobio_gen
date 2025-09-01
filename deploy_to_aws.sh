#!/bin/bash
# AWS Deployment Script for Astrobiology AI
# ==========================================
# One-click deployment script for complete AWS infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="astrobiology-ai"
AWS_REGION="us-west-2"
ENVIRONMENT="dev"
KEY_PAIR_NAME=""

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    print_success "AWS CLI is installed"
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    print_success "Terraform is installed"
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl is not installed. Kubernetes deployment will be skipped."
    else
        print_success "kubectl is installed"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials are not configured. Please run 'aws configure' first."
        exit 1
    fi
    print_success "AWS credentials are configured"
    
    # Get AWS account info
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_USER=$(aws sts get-caller-identity --query Arn --output text | cut -d'/' -f2)
    print_success "AWS Account: $AWS_ACCOUNT_ID, User: $AWS_USER"
}

setup_key_pair() {
    print_header "Setting up EC2 Key Pair"
    
    KEY_PAIR_NAME="${PROJECT_NAME}-keypair"
    
    # Check if key pair already exists
    if aws ec2 describe-key-pairs --key-names "$KEY_PAIR_NAME" --region "$AWS_REGION" &> /dev/null; then
        print_success "Key pair '$KEY_PAIR_NAME' already exists"
    else
        print_warning "Creating new key pair '$KEY_PAIR_NAME'"
        
        # Create key pair and save private key
        aws ec2 create-key-pair \
            --key-name "$KEY_PAIR_NAME" \
            --region "$AWS_REGION" \
            --query 'KeyMaterial' \
            --output text > "${KEY_PAIR_NAME}.pem"
        
        chmod 400 "${KEY_PAIR_NAME}.pem"
        print_success "Key pair created and saved as '${KEY_PAIR_NAME}.pem'"
        print_warning "Keep this file safe! You'll need it to access your instances."
    fi
}

deploy_infrastructure() {
    print_header "Deploying AWS Infrastructure with Terraform"
    
    cd terraform
    
    # Initialize Terraform
    print_warning "Initializing Terraform..."
    terraform init
    
    # Create terraform.tfvars file
    cat > terraform.tfvars << EOF
aws_region      = "$AWS_REGION"
project_name    = "$PROJECT_NAME"
environment     = "$ENVIRONMENT"
key_pair_name   = "$KEY_PAIR_NAME"
EOF
    
    # Plan deployment
    print_warning "Planning Terraform deployment..."
    terraform plan -var-file=terraform.tfvars
    
    # Ask for confirmation
    echo -e "${YELLOW}"
    read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
    echo -e "${NC}"
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Apply deployment
        print_warning "Deploying infrastructure..."
        terraform apply -var-file=terraform.tfvars -auto-approve
        
        # Get outputs
        VPC_ID=$(terraform output -raw vpc_id)
        DATA_BUCKET=$(terraform output -raw data_bucket_name)
        MODEL_BUCKET=$(terraform output -raw model_bucket_name)
        
        print_success "Infrastructure deployed successfully!"
        print_success "VPC ID: $VPC_ID"
        print_success "Data Bucket: $DATA_BUCKET"
        print_success "Model Bucket: $MODEL_BUCKET"
        
        # Save outputs to file
        cat > ../deployment_outputs.txt << EOF
VPC_ID=$VPC_ID
DATA_BUCKET=$DATA_BUCKET
MODEL_BUCKET=$MODEL_BUCKET
KEY_PAIR_NAME=$KEY_PAIR_NAME
AWS_REGION=$AWS_REGION
EOF
        
    else
        print_warning "Deployment cancelled by user"
        exit 0
    fi
    
    cd ..
}

upload_training_code() {
    print_header "Uploading Training Code to S3"
    
    # Load deployment outputs
    source deployment_outputs.txt
    
    # Create deployment package
    print_warning "Creating deployment package..."
    
    # Create temporary directory
    mkdir -p temp_deployment
    
    # Copy necessary files
    cp -r models/ temp_deployment/
    cp -r training/ temp_deployment/
    cp -r config/ temp_deployment/
    cp *.py temp_deployment/
    cp requirements.txt temp_deployment/ 2>/dev/null || echo "requirements.txt not found, creating one..."
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f temp_deployment/requirements.txt ]; then
        cat > temp_deployment/requirements.txt << EOF
torch>=2.0.0
torchvision
torchaudio
transformers>=4.20.0
diffusers>=0.20.0
torch-geometric
wandb
boto3
pyyaml
numpy
pandas
matplotlib
seaborn
jupyter
tensorboard
scikit-learn
scipy
networkx
pyro-ppl
dowhy
EOF
    fi
    
    # Create deployment archive
    cd temp_deployment
    zip -r ../astrobio-training-code.zip .
    cd ..
    
    # Upload to S3
    print_warning "Uploading training code to S3..."
    aws s3 cp astrobio-training-code.zip s3://$DATA_BUCKET/code/
    
    # Clean up
    rm -rf temp_deployment astrobio-training-code.zip
    
    print_success "Training code uploaded to S3"
}

setup_data_pipeline() {
    print_header "Setting up Data Pipeline"
    
    # Load deployment outputs
    source deployment_outputs.txt
    
    # Create sample data structure in S3
    print_warning "Setting up S3 data structure..."
    
    # Create directories in S3
    aws s3api put-object --bucket $DATA_BUCKET --key raw_data/ --region $AWS_REGION
    aws s3api put-object --bucket $DATA_BUCKET --key processed_data/ --region $AWS_REGION
    aws s3api put-object --bucket $DATA_BUCKET --key feature_store/ --region $AWS_REGION
    aws s3api put-object --bucket $MODEL_BUCKET --key checkpoints/ --region $AWS_REGION
    aws s3api put-object --bucket $MODEL_BUCKET --key logs/ --region $AWS_REGION
    aws s3api put-object --bucket $MODEL_BUCKET --key artifacts/ --region $AWS_REGION
    
    print_success "S3 data structure created"
    
    # Create data sync script
    cat > sync_local_data.sh << EOF
#!/bin/bash
# Sync local data to S3
echo "Syncing local data to S3..."

# Upload any local data files
if [ -d "data/" ]; then
    aws s3 sync data/ s3://$DATA_BUCKET/raw_data/ --region $AWS_REGION
    echo "Local data synced to S3"
else
    echo "No local data directory found"
fi

# Upload any local models
if [ -d "checkpoints/" ]; then
    aws s3 sync checkpoints/ s3://$MODEL_BUCKET/checkpoints/ --region $AWS_REGION
    echo "Local checkpoints synced to S3"
else
    echo "No local checkpoints directory found"
fi
EOF
    
    chmod +x sync_local_data.sh
    print_success "Data sync script created: sync_local_data.sh"
}

start_training_instance() {
    print_header "Starting Training Instance"
    
    # Load deployment outputs
    source deployment_outputs.txt
    
    # Get Auto Scaling Group name
    ASG_NAME=$(aws autoscaling describe-auto-scaling-groups \
        --region $AWS_REGION \
        --query "AutoScalingGroups[?contains(AutoScalingGroupName, '$PROJECT_NAME')].AutoScalingGroupName" \
        --output text)
    
    if [ -z "$ASG_NAME" ]; then
        print_error "Auto Scaling Group not found"
        exit 1
    fi
    
    # Set desired capacity to 1 to start an instance
    print_warning "Starting training instance..."
    aws autoscaling set-desired-capacity \
        --auto-scaling-group-name $ASG_NAME \
        --desired-capacity 1 \
        --region $AWS_REGION
    
    print_success "Training instance is starting..."
    print_warning "It may take 5-10 minutes for the instance to be fully ready"
    
    # Wait for instance to be running
    print_warning "Waiting for instance to be running..."
    sleep 30
    
    # Get instance IP
    INSTANCE_ID=$(aws autoscaling describe-auto-scaling-groups \
        --auto-scaling-group-names $ASG_NAME \
        --region $AWS_REGION \
        --query "AutoScalingGroups[0].Instances[0].InstanceId" \
        --output text)
    
    if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
        INSTANCE_IP=$(aws ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --region $AWS_REGION \
            --query "Reservations[0].Instances[0].PublicIpAddress" \
            --output text)
        
        print_success "Training instance is running!"
        print_success "Instance ID: $INSTANCE_ID"
        print_success "Public IP: $INSTANCE_IP"
        print_success "SSH Command: ssh -i ${KEY_PAIR_NAME}.pem ec2-user@$INSTANCE_IP"
        print_success "Jupyter Notebook: http://$INSTANCE_IP:8888"
        print_success "TensorBoard: http://$INSTANCE_IP:6006"
        
        # Save connection info
        cat > connection_info.txt << EOF
INSTANCE_ID=$INSTANCE_ID
INSTANCE_IP=$INSTANCE_IP
SSH_COMMAND=ssh -i ${KEY_PAIR_NAME}.pem ec2-user@$INSTANCE_IP
JUPYTER_URL=http://$INSTANCE_IP:8888
TENSORBOARD_URL=http://$INSTANCE_IP:6006
EOF
        
    else
        print_warning "Instance is still starting up. Check AWS console for status."
    fi
}

show_next_steps() {
    print_header "Next Steps"
    
    echo -e "${GREEN}"
    echo "🎉 AWS deployment completed successfully!"
    echo ""
    echo "📋 What was deployed:"
    echo "   ✅ VPC with public/private subnets"
    echo "   ✅ S3 buckets for data and models"
    echo "   ✅ EC2 training instances with GPU support"
    echo "   ✅ Auto Scaling Group for cost optimization"
    echo "   ✅ CloudWatch monitoring and logging"
    echo "   ✅ IAM roles and security groups"
    echo ""
    echo "🚀 To start training:"
    echo "   1. SSH into your training instance:"
    if [ -f connection_info.txt ]; then
        source connection_info.txt
        echo "      $SSH_COMMAND"
    fi
    echo ""
    echo "   2. Run the optimized training:"
    echo "      cd /home/ec2-user/astrobio-ai"
    echo "      python aws_optimized_training.py --mixed-precision --transfer-learning --progressive-training"
    echo ""
    echo "📊 Monitor training:"
    if [ -f connection_info.txt ]; then
        echo "   Jupyter Notebook: $JUPYTER_URL"
        echo "   TensorBoard: $TENSORBOARD_URL"
    fi
    echo "   AWS CloudWatch: https://console.aws.amazon.com/cloudwatch/"
    echo ""
    echo "💰 Cost optimization:"
    echo "   - Training instances auto-scale down when not in use"
    echo "   - Use spot instances for additional cost savings"
    echo "   - Monitor costs in AWS Cost Explorer"
    echo ""
    echo "🔧 Useful commands:"
    echo "   - Sync local data: ./sync_local_data.sh"
    echo "   - Scale up training: aws autoscaling set-desired-capacity --auto-scaling-group-name <ASG_NAME> --desired-capacity 3"
    echo "   - Scale down training: aws autoscaling set-desired-capacity --auto-scaling-group-name <ASG_NAME> --desired-capacity 0"
    echo -e "${NC}"
}

cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    
    if [ -d "terraform" ]; then
        cd terraform
        terraform destroy -var-file=terraform.tfvars -auto-approve 2>/dev/null || true
        cd ..
    fi
    
    # Clean up temporary files
    rm -f deployment_outputs.txt connection_info.txt sync_local_data.sh
    rm -rf temp_deployment
    
    print_warning "Cleanup completed. You can retry the deployment."
}

# Main execution
main() {
    print_header "AWS Deployment for Astrobiology AI"
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Run deployment steps
    check_prerequisites
    setup_key_pair
    deploy_infrastructure
    upload_training_code
    setup_data_pipeline
    start_training_instance
    show_next_steps
    
    print_success "Deployment completed successfully! 🎉"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--region REGION] [--environment ENV] [--project-name NAME]"
            echo ""
            echo "Options:"
            echo "  --region         AWS region (default: us-west-2)"
            echo "  --environment    Environment name (default: dev)"
            echo "  --project-name   Project name (default: astrobiology-ai)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
