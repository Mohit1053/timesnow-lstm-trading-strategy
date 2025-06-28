@echo off
echo 🔑 Setting up SSH keys for Kaggle remote access
echo ================================================

REM Check if .ssh directory exists
if not exist "%USERPROFILE%\.ssh" (
    echo Creating .ssh directory...
    mkdir "%USERPROFILE%\.ssh"
)

echo.
echo 📍 Current directory: %USERPROFILE%\.ssh
echo.

REM Generate SSH key
echo 🔨 Generating SSH key pair...
echo.
echo Press ENTER for all prompts (default location and no passphrase)
echo.
ssh-keygen -t rsa -f "%USERPROFILE%\.ssh\id_rsa"

REM Check if key was generated successfully
if exist "%USERPROFILE%\.ssh\id_rsa.pub" (
    echo.
    echo ✅ SSH keys generated successfully!
    echo.
    echo 📂 Files created:
    echo   - Private key: %USERPROFILE%\.ssh\id_rsa
    echo   - Public key:  %USERPROFILE%\.ssh\id_rsa.pub
    echo.
    
    REM Create authorized_keys file
    echo 📋 Creating authorized_keys file for GitHub upload...
    copy "%USERPROFILE%\.ssh\id_rsa.pub" "%USERPROFILE%\.ssh\authorized_keys"
    
    echo.
    echo ✅ Setup complete!
    echo.
    echo 📋 NEXT STEPS:
    echo 1. Upload the file 'authorized_keys' to a PUBLIC GitHub repository
    echo 2. Get the RAW link to that file
    echo 3. Copy your Ngrok authtoken from https://ngrok.com/
    echo 4. Use the Kaggle notebook template we provide
    echo.
    echo 📁 Your authorized_keys file location:
    echo %USERPROFILE%\.ssh\authorized_keys
    echo.
    echo Opening SSH directory...
    explorer "%USERPROFILE%\.ssh"
    
) else (
    echo.
    echo ❌ Failed to generate SSH keys. Please check your setup.
    echo.
)

echo.
echo Press any key to exit...
pause >nul
