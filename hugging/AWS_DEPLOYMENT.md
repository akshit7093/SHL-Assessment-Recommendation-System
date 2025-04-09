# AWS EC2 Deployment Guide

## Prerequisites
1. AWS Account with billing set up
2. AWS CLI installed locally
3. SSH client (PuTTY for Windows or native SSH for Unix-based systems)

## Step 1: Launch EC2 Instance
1. Choose Amazon Linux 2 AMI
2. Select t2.micro instance type (free tier eligible)
3. Configure Security Group:
   - Allow SSH (Port 22)
   - Allow HTTP (Port 80)
   - Allow HTTPS (Port 443)
   - Allow Custom TCP (Port 5000) for WebSocket

## Step 2: Connect to Instance
```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

## Step 3: Install Dependencies
```bash
# Update system packages
sudo yum update -y

# Install Python 3 and development tools
sudo yum install python3-pip python3-devel gcc git -y

# Install required system libraries for Vosk
sudo yum install wget unzip -y
```

## Step 4: Set Up Application
```bash
# Create application directory
mkdir ~/lang_convert
cd ~/lang_convert

# Clone the repository
git clone https://github.com/akshit7093/lang_convert.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Download Vosk model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

## Step 5: Configure Systemd Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/lang_convert.service
```

Add the following content:
```ini
[Unit]
Description=Language Conversion Web Application
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/lang_convert
Environment="PATH=/home/ec2-user/lang_convert/venv/bin"
ExecStart=/home/ec2-user/lang_convert/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Step 6: Start and Enable Service
```bash
# Reload systemd
sudo systemctl daemon-reload

sudo systemctl start lang_convert
sudo systemctl restart lang_convert

# Start service


# Enable service to start on boot
sudo systemctl enable lang_convert
```

## Step 7: Monitor Application
```bash
# Check service status
sudo systemctl status lang_convert

# View logs
journalctl -u lang_convert -f
```

## Step 8: Setting Up a Permanent URL

### Option 1: Using Amazon Route 53 (Recommended)
1. Register a domain through Route 53 or transfer existing domain
2. Create a hosted zone in Route 53
3. Create an A record pointing to your EC2 instance:
   - Name: your domain (e.g., example.com)
   - Type: A - IPv4 address
   - Value: Your EC2 instance's public IP
   - TTL: 300 seconds

### Option 2: Using Custom Domain from Another Registrar
1. Log in to your domain registrar
2. Update nameservers or create DNS records:
   - Type: A record
   - Host: @ or subdomain
   - Points to: Your EC2 instance's public IP
   - TTL: 300 seconds

### Setting up SSL/TLS Certificate
1. Using AWS Certificate Manager (ACM):
   ```bash
   # Install NGINX
   sudo yum install nginx -y
   
   # Request certificate in ACM console
   # Configure NGINX with SSL
   sudo nano /etc/nginx/conf.d/your-domain.conf
   ```

2. Using Let's Encrypt:
   ```bash
   # Install certbot
   sudo yum install certbot python3-certbot-nginx -y
   
   # Obtain certificate
   sudo certbot --nginx -d your-domain.com
   ```

## Cost Optimization Tips
1. Use t2.micro instance (free tier eligible)
2. Monitor CPU credits and usage patterns
3. Set up CloudWatch alarms for cost monitoring
4. Consider using Elastic IP only if necessary (additional cost)
5. Choose domain registration period wisely

## Security Recommendations
1. Keep security groups restrictive
2. Regularly update system packages
3. Use SSL/TLS for production (Let's Encrypt)
4. Implement rate limiting
5. Back up Vosk model and application data

## Troubleshooting
1. Check application logs: `journalctl -u lang_convert -f`
2. Verify permissions: `ls -la ~/lang_convert`
3. Monitor system resources: `top` or `htop`
4. Check network connectivity: `netstat -tulpn`