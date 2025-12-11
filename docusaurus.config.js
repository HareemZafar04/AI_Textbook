// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to type-check this file
// even if it doesn't use TypeScript.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI Textbook',
  tagline: 'Comprehensive Guide to Artificial Intelligence',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://ai-robotics-book.vercel.app',  // Updated for Vercel deployment
  baseUrl: '/',
  organizationName: 'ai-textbook',
  projectName: 'ai-textbook',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn', // Keep this for now to avoid breaking changes
  markdown: {
    format: 'detect',
    mermaid: false,
    mdx1Compat: {
      comments: true,
      admonitions: true,
      headingIds: true,
    },
  },

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
            'https://github.com/HareemZafar04/ai_robotics_book/edit/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/HareemZafar04/ai_robotics_book/edit/main/',
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
          { to: '/quiz', label: 'Quiz', position: 'left' },
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://github.com/HareemZafar04/ai_robotics_book',
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
                href: 'https://community.ai-textbook.org',  // Placeholder - replace with actual community link
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
                href: 'https://github.com/HareemZafar04/ai_robotics_book',
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