---
sidebar_label: Installation
---

# Installation

This guide will help you set up the tools and environment needed to work with AI concepts covered in this textbook.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Node.js (for running the documentation locally)
- Git for version control

## Setting up Python Environment

We recommend using a virtual environment to manage dependencies:

```bash
python -m venv ai-textbook-env
source ai-textbook-env/bin/activate  # On Windows: ai-textbook-env\Scripts\activate
pip install --upgrade pip
```

## Required Python Packages

Install the core AI libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch
```

## Running Documentation Locally

To run this documentation site locally:

```bash
npm install
npm start
```

This will start a local development server at `http://localhost:3000`.

## Verifying Installation

To verify that your installation is working correctly, run:

```bash
python -c "import numpy; import pandas; import sklearn; print('AI environment ready!')"
```

You should see the message "AI environment ready!" printed to the console.

## Next Steps

Once you have your environment set up, we recommend proceeding to the [Quick Start](quick-start.md) guide to begin experimenting with basic AI concepts.