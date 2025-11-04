# GitHub Upload Guide

## 🎯 Quick Start (3 Methods)

### **Method 1: Using Batch File (Easiest)**

1. Create a new repository on GitHub: https://github.com/new
2. Run the batch file:
   ```bash
   .\push_to_github.bat
   ```
3. Follow the prompts

---

### **Method 2: Manual Commands**

```bash
# 1. Initialize Git
git init

# 2. Add all files
git add .

# 3. Create first commit
git commit -m "Initial commit: Toxic Comment Classifier"

# 4. Add remote (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

---

### **Method 3: GitHub Desktop (GUI)**

1. Download GitHub Desktop: https://desktop.github.com/
2. Install and sign in
3. File → Add Local Repository
4. Select your project folder
5. Click "Publish repository"

---

## 📋 **Detailed Step-by-Step**

### **Step 1: Install Git**

**Check if Git is installed:**
```bash
git --version
```

**If not installed:**
- Download: https://git-scm.com/downloads
- Install with default settings
- Restart Command Prompt

---

### **Step 2: Configure Git (First Time Only)**

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify:**
```bash
git config --list
```

---

### **Step 3: Create GitHub Repository**

1. Go to: https://github.com/new
2. Fill in:
   - **Repository name**: `Toxic-Comment-Classifier`
   - **Description**: "NLP-based toxic comment classifier using Naive Bayes, Logistic Regression, and LSTM"
   - **Visibility**: Public (recommended) or Private
   - **DO NOT** initialize with README, .gitignore, or license
3. Click **"Create repository"**

---

### **Step 4: Initialize Local Repository**

```bash
cd "c:\xampp\htdocs\Toxic Comment Classifier"
git init
```

---

### **Step 5: Stage Files**

```bash
# Add all files
git add .

# Or add specific files
git add README.md
git add app.py
git add requirements.txt
```

**Check what will be committed:**
```bash
git status
```

---

### **Step 6: Create Commit**

```bash
git commit -m "Initial commit: Toxic Comment Classifier with NB, LR, and LSTM"
```

**Good commit messages:**
- "Initial commit: Complete toxic comment classifier"
- "Add: Naive Bayes, Logistic Regression, and LSTM models"
- "Feature: Streamlit web interface for predictions"

---

### **Step 7: Connect to GitHub**

```bash
# Add remote (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

---

### **Step 8: Authentication**

**Option A: Personal Access Token (Recommended)**

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "Toxic Comment Classifier"
4. Expiration: 90 days (or custom)
5. Select scopes: ✅ `repo` (full control)
6. Click "Generate token"
7. **Copy the token** (save it somewhere safe!)
8. When Git asks for password, paste the token

**Option B: GitHub CLI**

```bash
# Install GitHub CLI
winget install --id GitHub.cli

# Authenticate
gh auth login

# Push
git push -u origin main
```

**Option C: SSH Key**

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
# Then use SSH URL
git remote set-url origin git@github.com:YOUR-USERNAME/Toxic-Comment-Classifier.git
```

---

## 🔄 **Future Updates**

After making changes:

```bash
# 1. Check what changed
git status

# 2. Add changes
git add .

# 3. Commit with message
git commit -m "Update: Description of changes"

# 4. Push to GitHub
git push
```

**Or use the batch file:**
```bash
.\push_to_github.bat
```

---

## 📁 **What Gets Uploaded**

The `.gitignore` file prevents these from being uploaded:
- ✅ Python cache (`__pycache__/`)
- ✅ Virtual environments (`venv/`, `env/`)
- ✅ Large data files (`data/*.csv`)
- ✅ Trained models (`models/*.pkl`, `models/*.h5`)
- ✅ IDE settings (`.vscode/`, `.idea/`)

**What WILL be uploaded:**
- ✅ Source code (`src/`, `app.py`)
- ✅ Documentation (`.md` files)
- ✅ Configuration (`requirements.txt`, `config.py`)
- ✅ Scripts (`.py`, `.bat` files)
- ✅ Notebook (`experiment.ipynb`)

---

## 💡 **Best Practices**

### **Commit Messages**

Good:
```bash
git commit -m "Add: LSTM model with bidirectional layers"
git commit -m "Fix: Preprocessing bug with special characters"
git commit -m "Update: Improve web UI styling"
git commit -m "Docs: Add installation guide"
```

Bad:
```bash
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

### **Commit Frequency**

- Commit after completing a feature
- Commit before making major changes
- Commit at the end of each work session
- Don't commit broken code

### **Branch Strategy**

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes and commit
git add .
git commit -m "Add: BERT model implementation"

# Push branch
git push -u origin feature/new-model

# Merge on GitHub via Pull Request
```

---

## 🔍 **Useful Git Commands**

```bash
# View commit history
git log

# View changes
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- filename

# Update from GitHub
git pull

# Clone repository
git clone https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier.git

# View remotes
git remote -v

# Create branch
git branch feature-name

# Switch branch
git checkout feature-name

# Merge branch
git merge feature-name

# Delete branch
git branch -d feature-name
```

---

## 🐛 **Troubleshooting**

### **Issue: "fatal: not a git repository"**
```bash
git init
```

### **Issue: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier.git
```

### **Issue: "failed to push some refs"**
```bash
# Pull first, then push
git pull origin main --rebase
git push origin main
```

### **Issue: "Authentication failed"**
- Use Personal Access Token, not password
- Generate new token: https://github.com/settings/tokens

### **Issue: "Large files"**
```bash
# Remove large files from tracking
git rm --cached data/large_file.csv
git commit -m "Remove large file"
```

---

## 📊 **GitHub Repository Setup**

### **Add Topics**

On GitHub, click "⚙️ Settings" → "Topics":
- `machine-learning`
- `nlp`
- `toxic-comment-classification`
- `deep-learning`
- `lstm`
- `streamlit`
- `python`
- `text-classification`

### **Add Description**

"🛡️ NLP-based toxic comment classifier using Naive Bayes, Logistic Regression, and LSTM with interactive Streamlit web interface"

### **Add Website**

If you deploy to Streamlit Cloud, add the URL

### **Enable Issues**

Settings → Features → ✅ Issues

### **Add License**

Add file → Create new file → Name: `LICENSE`
- Choose MIT License or Apache 2.0

---

## 🚀 **Deploy to Streamlit Cloud**

1. Push to GitHub (follow steps above)
2. Go to: https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Main file: `app.py`
7. Click "Deploy"

Your app will be live at: `https://YOUR-USERNAME-toxic-comment-classifier.streamlit.app`

---

## 📝 **Repository URL**

After uploading, your repository will be at:

```
https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier
```

Share this link to showcase your project!

---

## ✅ **Checklist**

Before pushing to GitHub:

- [ ] Git installed and configured
- [ ] GitHub account created
- [ ] Repository created on GitHub
- [ ] All files committed locally
- [ ] `.gitignore` file present
- [ ] README.md is complete
- [ ] No sensitive data (API keys, passwords)
- [ ] Large files excluded
- [ ] Code is working
- [ ] Documentation is clear

---

## 🎓 **Learning Resources**

- **Git Basics**: https://git-scm.com/book/en/v2
- **GitHub Guides**: https://guides.github.com/
- **Interactive Tutorial**: https://learngitbranching.js.org/
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf

---

**Need help?** Check the troubleshooting section or visit: https://docs.github.com/
