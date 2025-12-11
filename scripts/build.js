#!/usr/bin/env node

// Custom build script to avoid binary permission issues on Vercel
const { build } = require('@docusaurus/core/lib/commands/build/build.js');
const config = require('../docusaurus.config.js');

// Perform the build
build(process.cwd(), config)
  .then(() => {
    console.log('Build completed successfully');
    process.exit(0);
  })
  .catch((err) => {
    console.error('Build failed:', err);
    process.exit(1);
  });