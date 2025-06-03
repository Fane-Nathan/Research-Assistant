# GitHub Pages Setup Guide for Docusaurus Documentation

This guide will help you deploy your Docusaurus documentation to GitHub Pages.

## 📋 Prerequisites

- [x] GitHub repository: `Fane-Nathan/Research-Assistant`
- [x] Docusaurus site configured in `docs/` directory
- [x] GitHub Actions workflow created

## 🚀 Deployment Steps

### Step 1: Enable GitHub Pages

1. **Navigate to Repository Settings**
   - Go to your repository: https://github.com/Fane-Nathan/Research-Assistant
   - Click on the **Settings** tab

2. **Configure GitHub Pages**
   - Scroll down to **Pages** in the left sidebar
   - Under **Source**, select **"GitHub Actions"**
   - This enables deployment via GitHub Actions workflow

### Step 2: Verify Configuration

Your repository is already configured with:

✅ **Docusaurus Config** (`docs/docusaurus.config.ts`):
```typescript
url: 'https://fane-nathan.github.io',
baseUrl: '/Research-Assistant/',
organizationName: 'Fane-Nathan',
projectName: 'Research-Assistant',
```

✅ **GitHub Actions Workflow** (`.github/workflows/deploy-docs.yml`):
- Automatically builds and deploys on pushes to `main` branch
- Triggers on changes to `docs/**` directory
- Can be manually triggered from Actions tab

### Step 3: Deploy

1. **Commit and Push Changes**
   ```bash
   git add .
   git commit -m "Setup GitHub Pages deployment for documentation"
   git push origin main
   ```

2. **Monitor Deployment**
   - Go to **Actions** tab in your repository
   - Watch the "Deploy Docusaurus to GitHub Pages" workflow
   - First deployment may take 2-5 minutes

3. **Access Your Documentation**
   - Once deployed, visit: **https://fane-nathan.github.io/Research-Assistant/**
   - The site will update automatically on future pushes to main

## 🔧 Workflow Features

### Automatic Deployment
- **Triggers**: Pushes to `main` branch affecting `docs/**` files
- **Manual**: Can be triggered from GitHub Actions tab
- **Builds**: Uses Node.js 20 with npm caching for faster builds

### Security & Permissions
- Uses minimal required permissions
- Prevents concurrent deployments to avoid conflicts
- Secure deployment to GitHub Pages environment

## 🛠️ Local Development

To test documentation locally before deployment:

```bash
cd docs
npm install
npm start
```

This will start a local development server at http://localhost:3000

## 📁 Project Structure

```
Research-Assistant/
├── docs/                           # Docusaurus documentation
│   ├── docs/                      # Documentation pages
│   ├── blog/                      # Blog posts (optional)
│   ├── src/                       # Custom React components
│   ├── static/                    # Static assets
│   ├── docusaurus.config.ts       # Docusaurus configuration
│   ├── package.json               # Dependencies
│   └── sidebars.ts                # Navigation sidebar
├── .github/
│   └── workflows/
│       └── deploy-docs.yml         # GitHub Actions workflow
└── [other project files]
```

## 🔍 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check Actions tab for detailed error logs
   - Ensure all dependencies are in `docs/package.json`
   - Verify Node.js compatibility

2. **404 Page Not Found**
   - Verify `baseUrl` in `docusaurus.config.ts` matches repository name
   - Check that GitHub Pages source is set to "GitHub Actions"

3. **CSS/Assets Not Loading**
   - Ensure `baseUrl` is correctly set to `/Research-Assistant/`
   - Check that asset paths are relative

### Manual Verification

To manually trigger deployment:
1. Go to repository **Actions** tab
2. Select "Deploy Docusaurus to GitHub Pages" workflow  
3. Click **Run workflow** → **Run workflow**

## 📞 Next Steps

After successful deployment:

1. **Custom Domain** (optional): Add a CNAME file to `docs/static/` directory
2. **SEO Optimization**: Update metadata in `docusaurus.config.ts`
3. **Analytics**: Add Google Analytics or other tracking
4. **Content**: Continue adding documentation in `docs/docs/` directory

## 🎉 Success!

Once deployed, your documentation will be available at:
**https://fane-nathan.github.io/Research-Assistant/**

The site will automatically update whenever you push documentation changes to the main branch.
