#!/bin/bash
# EC2 User Data Script for Astrobiology AI Training
# ==================================================
# Automated setup script for training instances

set -e

# Variables from Terraform
S3_DATA_BUCKET="${s3_data_bucket}"
S3_MODEL_BUCKET="${s3_model_bucket}"
AWS_REGION="${aws_region}"

# Logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting Astrobiology AI training instance setup..."

# Update system
echo "Updating system packages..."
yum update -y

# Install additional packages
echo "Installing additional packages..."
yum install -y \
    git \
    htop \
    tmux \
    vim \
    wget \
    curl \
    unzip \
    tree

# Install AWS CLI v2
echo "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Configure AWS CLI
echo "Configuring AWS CLI..."
aws configure set region $AWS_REGION
aws configure set output json

# Install Docker
echo "Installing Docker..."
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | tee /etc/yum.repos.d/nvidia-docker.repo
yum install -y nvidia-container-toolkit
systemctl restart docker

# Setup Python environment
echo "Setting up Python environment..."
# The Deep Learning AMI already has conda installed
source /home/ec2-user/anaconda3/bin/activate

# Create conda environment for astrobiology AI
echo "Creating conda environment..."
conda create -n astrobio python=3.9 -y
source activate astrobio

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional Python packages
echo "Installing additional Python packages..."
pip install \
    transformers \
    diffusers \
    torch-geometric \
    wandb \
    boto3 \
    pyyaml \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    tensorboard \
    scikit-learn \
    scipy \
    networkx \
    pyro-ppl \
    dowhy

# Install specific versions for compatibility
pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Setup project directory
echo "Setting up project directory..."
mkdir -p /home/ec2-user/astrobio-ai
cd /home/ec2-user/astrobio-ai

# Clone repository (replace with your actual repository)
echo "Cloning repository..."
# git clone https://github.com/your-username/astrobio-ai.git .

# Create necessary directories
mkdir -p data models checkpoints logs config

# Download data from S3
echo "Setting up data synchronization..."
cat > /home/ec2-user/sync_data.sh << EOF
#!/bin/bash
# Sync data from S3
aws s3 sync s3://$S3_DATA_BUCKET/raw_data/ /home/ec2-user/astrobio-ai/data/raw/
aws s3 sync s3://$S3_DATA_BUCKET/processed_data/ /home/ec2-user/astrobio-ai/data/processed/
aws s3 sync s3://$S3_MODEL_BUCKET/checkpoints/ /home/ec2-user/astrobio-ai/checkpoints/
EOF

chmod +x /home/ec2-user/sync_data.sh

# Setup model upload script
cat > /home/ec2-user/upload_models.sh << EOF
#!/bin/bash
# Upload trained models to S3
aws s3 sync /home/ec2-user/astrobio-ai/checkpoints/ s3://$S3_MODEL_BUCKET/checkpoints/
aws s3 sync /home/ec2-user/astrobio-ai/logs/ s3://$S3_MODEL_BUCKET/logs/
EOF

chmod +x /home/ec2-user/upload_models.sh

# Setup Jupyter notebook service
echo "Setting up Jupyter notebook service..."
cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/astrobio-ai
ExecStart=/home/ec2-user/anaconda3/envs/astrobio/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable jupyter
systemctl start jupyter

# Setup TensorBoard service
echo "Setting up TensorBoard service..."
cat > /etc/systemd/system/tensorboard.service << EOF
[Unit]
Description=TensorBoard Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/astrobio-ai
ExecStart=/home/ec2-user/anaconda3/envs/astrobio/bin/tensorboard --logdir=logs --host=0.0.0.0 --port=6006
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tensorboard
systemctl start tensorboard

# Setup CloudWatch agent
echo "Setting up CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# CloudWatch agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/astrobiology-ai-training",
                        "log_stream_name": "{instance_id}/user-data"
                    },
                    {
                        "file_path": "/home/ec2-user/astrobio-ai/logs/*.log",
                        "log_group_name": "/aws/ec2/astrobiology-ai-training",
                        "log_stream_name": "{instance_id}/training"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "AstrobiologyAI/EC2",
        "metrics_collected": {
            "cpu": {
                "measurement": ["cpu_usage_idle", "cpu_usage_iowait", "cpu_usage_user", "cpu_usage_system"],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": ["used_percent"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
            },
            "mem": {
                "measurement": ["mem_used_percent"],
                "metrics_collection_interval": 60
            },
            "nvidia_gpu": {
                "measurement": ["utilization_gpu", "utilization_memory", "temperature_gpu"],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Setup training environment variables
echo "Setting up environment variables..."
cat >> /home/ec2-user/.bashrc << EOF

# Astrobiology AI Environment
export ASTROBIO_HOME=/home/ec2-user/astrobio-ai
export S3_DATA_BUCKET=$S3_DATA_BUCKET
export S3_MODEL_BUCKET=$S3_MODEL_BUCKET
export AWS_DEFAULT_REGION=$AWS_REGION
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=\$ASTROBIO_HOME:\$PYTHONPATH

# Activate conda environment
source /home/ec2-user/anaconda3/bin/activate astrobio

# Aliases
alias ll='ls -la'
alias la='ls -la'
alias sync-data='/home/ec2-user/sync_data.sh'
alias upload-models='/home/ec2-user/upload_models.sh'
alias train-sota='cd \$ASTROBIO_HOME && python aws_optimized_training.py'
alias monitor-gpu='watch -n 1 nvidia-smi'

EOF

# Setup automatic data sync on startup
echo "Setting up automatic data sync..."
cat > /etc/systemd/system/data-sync.service << EOF
[Unit]
Description=Sync training data from S3
After=network.target

[Service]
Type=oneshot
User=ec2-user
ExecStart=/home/ec2-user/sync_data.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable data-sync
systemctl start data-sync

# Setup automatic model upload on shutdown
cat > /etc/systemd/system/model-upload.service << EOF
[Unit]
Description=Upload models to S3 on shutdown
DefaultDependencies=false
Before=shutdown.target reboot.target halt.target

[Service]
Type=oneshot
RemainAfterExit=true
User=ec2-user
ExecStart=/bin/true
ExecStop=/home/ec2-user/upload_models.sh
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable model-upload

# Set proper ownership
chown -R ec2-user:ec2-user /home/ec2-user/

# Create training status file
echo "Training instance ready" > /home/ec2-user/instance_status.txt
echo "Setup completed at: $(date)" >> /home/ec2-user/instance_status.txt

# Final message
echo "Astrobiology AI training instance setup completed successfully!"
echo "Instance is ready for training."
echo "Jupyter notebook available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ip):8888"
echo "TensorBoard available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ip):6006"

# Signal completion
/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource AutoScalingGroup --region ${AWS::Region} || true
