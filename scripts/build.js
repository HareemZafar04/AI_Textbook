#!/usr/bin/env node

// Custom build script to avoid binary permission issues on Vercel and fix CSS issues
const path = require('path');

// Set NODE_ENV to production
process.env.NODE_ENV = 'production';

// Import @docusaurus/core using dynamic require with error handling
let docusaurusCore;
try {
  docusaurusCore = require('@docusaurus/core');
} catch (error) {
  console.error('Error importing @docusaurus/core:', error.message);
  console.error('Make sure all dependencies are installed correctly.');
  console.error('Run `npm install` to install missing dependencies.');
  process.exit(1);
}

const build = docusaurusCore.build;

// Import config
let config;
try {
  config = require('../docusaurus.config.js');
} catch (error) {
  console.error('Error importing docusaurus.config.js:', error.message);
  process.exit(1);
}

// Add custom webpack config to handle CSS minimization
const updatedConfig = {
  ...config,
  webpack: {
    ...config.webpack,
    configureWebpack: (webpackConfig, isServer, utils) => {
      // Find and modify the CSS minimization plugin
      if (webpackConfig.optimization && webpackConfig.optimization.minimizer) {
        webpackConfig.optimization.minimizer = webpackConfig.optimization.minimizer
          .map(minimizer => {
            // If it's the CssMinimizerPlugin, modify its options
            if (minimizer.constructor.name === 'CssMinimizerPlugin') {
              try {
                const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
                return new CssMinimizerPlugin({
                  minimizerOptions: {
                    preset: [
                      'default',
                      {
                        // Disable SVGO to avoid the cleanupIds.js error
                        svgo: false,
                      },
                    ],
                  },
                });
              } catch (error) {
                console.warn('Warning: Could not import css-minimizer-webpack-plugin:', error.message);
                console.warn('Falling back to original minimizer.');
                return minimizer;
              }
            }
            return minimizer;
          });
      }

      // Call any existing webpack configuration function
      if (config.webpack && typeof config.webpack.configureWebpack === 'function') {
        return config.webpack.configureWebpack(webpackConfig, isServer, utils);
      }

      return webpackConfig;
    }
  }
};

// Perform the build
build(process.cwd(), updatedConfig)
  .then(() => {
    console.log('Build completed successfully');
    process.exit(0);
  })
  .catch((err) => {
    console.error('Build failed:', err);
    process.exit(1);
  });