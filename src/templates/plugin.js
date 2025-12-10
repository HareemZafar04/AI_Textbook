// @ts-check

/**
 * {pluginName} Plugin
 * 
 * This plugin provides functionality for ...
 */

/** @type {import('@docusaurus/types').PluginModule} */
module.exports = function pluginName(context, options) {
  return {
    name: '{pluginName}',

    // Example hook implementation
    // extendCli(cli) {
    //   cli
    //     .command('my-command')
    //     .description('My custom command')
    //     .action(() => {
    //       console.log('Running my custom command');
    //     });
    // },

    // Example configuration
    configureWebpack(config, isServer, utils) {
      return {
        module: {
          rules: [
            // Add your webpack rules here
          ],
        },
      };
    },

    // Add more plugin lifecycle hooks as needed
    // See: https://docusaurus.io/docs/using-plugins#creating-plugins
  };
};