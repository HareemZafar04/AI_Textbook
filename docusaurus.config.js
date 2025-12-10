// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to type-check this file
// even if it doesn't use TypeScript.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI Textbook',
  tagline: 'Comprehensive Guide to Artificial Intelligence',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'http://localhost',
  baseUrl: '/',
  organizationName: 'ai-textbook',
  projectName: 'ai-textbook',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/ai-textbook/ai-textbook/edit/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/ai-textbook/ai-textbook/edit/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'AI Textbook',
        logo: {
          alt: 'AI Textbook Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Documentation',
          },
          // ❌ REMOVED: {to: '/ai-textbook', label: 'AI Textbook', position: 'left'},
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://github.com/ai-textbook/ai-textbook',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
              {
                label: 'AI Community',
                href: 'https://example.com/ai-community',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/ai-textbook/ai-textbook',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI Textbook Project. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;
