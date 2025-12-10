import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', 'b5f'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '417'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'fa3'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'a26'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '088'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '8a9'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '0d3'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', 'e10'),
    exact: true
  },
  {
    path: '/blog/2023/01/01/welcome',
    component: ComponentCreator('/blog/2023/01/01/welcome', '5ac'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', 'e2b'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '1f9'),
    exact: true
  },
  {
    path: '/blog/tags/ai',
    component: ComponentCreator('/blog/tags/ai', 'c87'),
    exact: true
  },
  {
    path: '/blog/tags/education',
    component: ComponentCreator('/blog/tags/education', '952'),
    exact: true
  },
  {
    path: '/blog/tags/genai',
    component: ComponentCreator('/blog/tags/genai', '82c'),
    exact: true
  },
  {
    path: '/blog/tags/machine-learning',
    component: ComponentCreator('/blog/tags/machine-learning', '884'),
    exact: true
  },
  {
    path: '/blog/tags/research',
    component: ComponentCreator('/blog/tags/research', '395'),
    exact: true
  },
  {
    path: '/blog/tags/robotics',
    component: ComponentCreator('/blog/tags/robotics', 'b5e'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'dff'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '259'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'de5'),
            routes: [
              {
                path: '/docs/applications/autonomous-systems',
                component: ComponentCreator('/docs/applications/autonomous-systems', '824'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/finance',
                component: ComponentCreator('/docs/applications/finance', 'f70'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/healthcare',
                component: ComponentCreator('/docs/applications/healthcare', '52c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/robotics',
                component: ComponentCreator('/docs/applications/robotics', 'a3e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/applications',
                component: ComponentCreator('/docs/cv/applications', 'af9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/image-processing',
                component: ComponentCreator('/docs/cv/image-processing', 'a61'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/introduction',
                component: ComponentCreator('/docs/cv/introduction', 'b5a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/object-detection',
                component: ComponentCreator('/docs/cv/object-detection', '741'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/docs/intro',
                component: ComponentCreator('/docs/docs/intro', 'e37'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/ai-ethics',
                component: ComponentCreator('/docs/foundations/ai-ethics', 'e6d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/core-concepts',
                component: ComponentCreator('/docs/foundations/core-concepts', '3fa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/history',
                component: ComponentCreator('/docs/foundations/history', 'fa3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/types-of-ai',
                component: ComponentCreator('/docs/foundations/types-of-ai', '95e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '490'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/quick-start',
                component: ComponentCreator('/docs/getting-started/quick-start', 'c34'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/deep-learning',
                component: ComponentCreator('/docs/ml/deep-learning', '5c9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/introduction',
                component: ComponentCreator('/docs/ml/introduction', '359'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/reinforcement-learning',
                component: ComponentCreator('/docs/ml/reinforcement-learning', '74f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/supervised-learning',
                component: ComponentCreator('/docs/ml/supervised-learning', '88f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/unsupervised-learning',
                component: ComponentCreator('/docs/ml/unsupervised-learning', 'd93'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/applications',
                component: ComponentCreator('/docs/nlp/applications', '723'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/introduction',
                component: ComponentCreator('/docs/nlp/introduction', 'a3c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/language-models',
                component: ComponentCreator('/docs/nlp/language-models', 'f63'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/text-processing',
                component: ComponentCreator('/docs/nlp/text-processing', '0a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '9e4'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
