# 🚀 Remote SSH Kaggle + VS Code Setup Guide

This guide helps you set up remote SSH connection to Kaggle from VS Code, allowing you to run your LSTM trading strategy directly on Kaggle's environment with full VS Code features.

## 🎯 Benefits

✅ **12-hour continuous sessions** without interruptions  
✅ **Up to 42 hours GPU per week** (vs 30 hours default)  
✅ **Full VS Code features** on Kaggle environment  
✅ **Easy terminal access** and debugging  
✅ **File management** with .py files  
✅ **Free GPU acceleration** for your LSTM models  

## 📋 Prerequisites

1. **VS Code** installed
2. **Ngrok account** (free)
3. **GitHub account**
4. **Kaggle account**

## 🛠️ Step-by-Step Setup

### Step 1: Install Required Software

1. **Download VS Code**: https://code.visualstudio.com/
2. **Create Ngrok account**: https://ngrok.com/
3. **Install VS Code SSH Extensions**:
   - Open VS Code
   - Press `Ctrl+Shift+X`
   - Search and install:
     - `Remote - SSH`
     - `Remote - SSH: Editing Configuration Files`

### Step 2: Generate SSH Keys

#### Windows (Command Prompt/PowerShell):
```cmd
# Open Command Prompt or PowerShell
ssh-keygen -t rsa

# Press Enter for default location: C:\Users\YourName\.ssh\id_rsa
# Press Enter twice for no passphrase
```

#### Result Files:
- `id_rsa` (private key) - Keep secret!
- `id_rsa.pub` (public key) - Will upload to GitHub

### Step 3: Upload SSH Public Key to GitHub

1. **Navigate to your SSH directory**:
   - Windows: `C:\Users\YourName\.ssh\`
   
2. **Rename public key**:
   - Copy `id_rsa.pub` to `authorized_keys` (no extension)
   
3. **Create GitHub repository**:
   - Create new public repo: `SSH_Key_public`
   - Upload `authorized_keys` file
   - Make repository public

4. **Get raw link**:
   - Click on `authorized_keys` in GitHub
   - Click "Raw" button
   - Copy the URL (looks like: `https://raw.githubusercontent.com/yourusername/SSH_Key_public/main/authorized_keys`)

### Step 4: Get Ngrok Token

1. Go to https://ngrok.com/
2. Sign up/Login
3. Go to "Your Authtoken" section
4. Copy your authtoken (format: `2abc...xyz`)

### Step 5: Setup Kaggle Notebook

1. **Copy our prepared notebook** (we'll create this)
2. **Configure settings**:
   - Choose GPU (T4 x2 or P100)
   - Set Persistence to "Files only"
   - Turn on Internet access

3. **Set variables**:
   - `public_key_path`: Your GitHub raw URL from Step 3
   - `ngrok_token`: Your token from Step 4

### Step 6: Configure VS Code SSH

1. **Open VS Code**
2. **Press `Ctrl+Shift+P`**
3. **Type**: `Remote-SSH: Connect to Host...`
4. **Click**: `Configure SSH Hosts...`
5. **Select**: `~/.ssh/config` (first option)

6. **Add configuration**:
```
Host KaggleSSH
    HostName 0.tcp.ap.ngrok.io
    Port 12345
    User root
    IdentityFile ~/.ssh/id_rsa
```

*Note: HostName and Port will come from Kaggle notebook output*

### Step 7: Connect and Use

1. **Run Kaggle notebook** (get HostName and Port)
2. **Update VS Code config** with actual HostName and Port
3. **Connect**: `Ctrl+Shift+P` → `Remote-SSH: Connect to Host...` → `KaggleSSH`
4. **Open folder**: `/kaggle/working/`
5. **Activate environment**:
   ```bash
   conda init
   # Kill terminal and restart
   sudo apt install nvidia-utils-515 -y
   nvidia-smi  # Check GPU
   ```

## 🎮 Usage Tips

### Folder Structure in Kaggle:
- `/kaggle/input/` - Read-only data (107GB limit)
- `/kaggle/working/` - Your workspace (20GB limit)

### Extending GPU Time:
1. Use normal session for 29 hours
2. Close notebook before hitting 30-hour limit
3. SSH back in for additional 12 hours
4. Total: 42 hours per week!

### Running Your LSTM Strategy:
```bash
# Upload your script to /kaggle/working/
cd /kaggle/working/
python kaggle_lstm_trading_strategy.py
```

## 🔧 Troubleshooting

**Connection refused**: Check if Kaggle notebook is running and ngrok tunnel is active  
**Key permission denied**: Ensure SSH key permissions are correct  
**GPU not found**: Run `sudo apt install nvidia-utils-515 -y`  
**Import errors**: Use `pip install package-name` in Kaggle environment  

## 📁 Files We'll Create

1. `kaggle_ssh_notebook.ipynb` - Ready-to-use Kaggle notebook
2. `setup_ssh_keys.bat` - Windows script to generate keys  
3. `vscode_config_template.txt` - SSH config template
4. `test_connection.py` - Script to test your setup

## 🚀 Next Steps

After setup, you can:
- Run your LSTM trading strategy with GPU acceleration
- Use VS Code debugging features
- Manage your files easily
- Access full terminal capabilities
- Work with large datasets efficiently

Ready to get started? Follow the detailed setup in the next files!
