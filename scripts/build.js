#!/usr/bin/env node

// Custom build script to avoid binary permission issues on Vercel and fix CSS issues
const path = require('path');
const fs = require('fs');

// Set NODE_ENV to production
process.env.NODE_ENV = 'production';

// Dynamically import @docusaurus/core
const docusaurusCore = require('@docusaurus/core');
const build = docusaurusCore.build;

// Modify the config to fix CSS optimization issues
let config = require('../docusaurus.config.js');

// Add custom webpack config to handle CSS minimization
config = {
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
              // Create new instance of CssMinimizerPlugin with SVGO disabled
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
build(process.cwd(), config)
  .then(() => {
    console.log('Build completed successfully');
    process.exit(0);
  })
  .catch((err) => {
    console.error('Build failed:', err);
    process.exit(1);
  });