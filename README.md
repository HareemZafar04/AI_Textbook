# AI Textbook - Docusaurus Documentation Site

This project contains a Docusaurus-based documentation site for the AI Textbook project. The site serves as a comprehensive guide to artificial intelligence concepts, techniques, and applications.

## Prerequisites

Before running this project, make sure you have:

- Node.js version 18 or higher
- npm or yarn package manager
- Git for version control

## Installation

1. Clone the repository (if not already done):

```bash
git clone https://github.com/ai-textbook/ai-textbook.git
cd ai-textbook
```

2. Install dependencies:

```bash
npm install
```

## Local Development

To start the development server:

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build the Static Site

To build the static site for production:

```bash
npm run build
```

The static files will be generated in the `build/` directory and can be served using any static hosting service.

## Directory Structure

- `blog/` - Contains blog posts in markdown format
- `docs/` - Contains documentation files organized by topic
- `src/` - Contains custom React components and CSS
- `static/` - Contains static assets like images and favicons
- `docusaurus.config.js` - Main Docusaurus configuration file
- `sidebars.js` - Navigation sidebar configuration
- `package.json` - Project dependencies and scripts

## Adding New Content

### Adding a New Doc

1. Create a new markdown file in the appropriate subdirectory of `docs/`
2. Add the document to the sidebar in `sidebars.js`

Example markdown header:
```markdown
---
sidebar_label: Title of Document
---

# Title of Document

Content goes here...
```

### Adding a New Blog Post

1. Create a new markdown file in the `blog/` directory
2. Use the following format for the filename: `YYYY-MM-DD-post-title.md`

Example blog post header:
```markdown
---
title: Title of Blog Post
authors: [author1, author2]
tags: [tag1, tag2]
---

# Title of Blog Post

Content goes here...
```

## Configuration

The site configuration is stored in `docusaurus.config.js`. You can customize:

- Site metadata (title, tagline, description)
- Theme configuration
- Navigation links
- Social media links
- Deployment settings

## Deployment

To deploy the site:

1. Build the site: `npm run build`
2. Upload the contents of the `build/` directory to your web server

For deployment to GitHub Pages, refer to the [Docusaurus deployment guide](https://docusaurus.io/docs/deployment).

## Contributing

We welcome contributions to the AI Textbook project! If you'd like to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Add your changes and commit them with descriptive commit messages
4. Submit a pull request with a clear description of your changes

For major changes, please open an issue first to discuss what you would like to change."# ai_humanoid_robotics_course" 
"# ai_humanoid_robotics_book" 
