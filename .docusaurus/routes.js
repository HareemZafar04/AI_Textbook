import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/ai-textbook/__docusaurus/debug',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug', '24c'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/config',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/config', 'aa8'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/content',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/content', '622'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/globalData',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/globalData', '666'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/metadata',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/metadata', 'f1b'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/registry',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/registry', '464'),
    exact: true
  },
  {
    path: '/ai-textbook/__docusaurus/debug/routes',
    component: ComponentCreator('/ai-textbook/__docusaurus/debug/routes', 'd30'),
    exact: true
  },
  {
    path: '/ai-textbook/blog',
    component: ComponentCreator('/ai-textbook/blog', '0ad'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/2023/01/01/welcome',
    component: ComponentCreator('/ai-textbook/blog/2023/01/01/welcome', '349'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/archive',
    component: ComponentCreator('/ai-textbook/blog/archive', '362'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags',
    component: ComponentCreator('/ai-textbook/blog/tags', '53d'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/ai',
    component: ComponentCreator('/ai-textbook/blog/tags/ai', '7d0'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/education',
    component: ComponentCreator('/ai-textbook/blog/tags/education', '111'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/genai',
    component: ComponentCreator('/ai-textbook/blog/tags/genai', '81e'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/machine-learning',
    component: ComponentCreator('/ai-textbook/blog/tags/machine-learning', 'd73'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/research',
    component: ComponentCreator('/ai-textbook/blog/tags/research', '3ad'),
    exact: true
  },
  {
    path: '/ai-textbook/blog/tags/robotics',
    component: ComponentCreator('/ai-textbook/blog/tags/robotics', 'fcf'),
    exact: true
  },
  {
    path: '/ai-textbook/quiz',
    component: ComponentCreator('/ai-textbook/quiz', '7cf'),
    exact: true
  },
  {
    path: '/ai-textbook/docs',
    component: ComponentCreator('/ai-textbook/docs', 'c9f'),
    routes: [
      {
        path: '/ai-textbook/docs',
        component: ComponentCreator('/ai-textbook/docs', 'e96'),
        routes: [
          {
            path: '/ai-textbook/docs',
            component: ComponentCreator('/ai-textbook/docs', 'b2f'),
            routes: [
              {
                path: '/ai-textbook/docs/applications/autonomous-systems',
                component: ComponentCreator('/ai-textbook/docs/applications/autonomous-systems', '3fc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/applications/finance',
                component: ComponentCreator('/ai-textbook/docs/applications/finance', 'b58'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/applications/healthcare',
                component: ComponentCreator('/ai-textbook/docs/applications/healthcare', 'bc5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/applications/robotics',
                component: ComponentCreator('/ai-textbook/docs/applications/robotics', 'a0e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/cv/applications',
                component: ComponentCreator('/ai-textbook/docs/cv/applications', '319'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/cv/image-processing',
                component: ComponentCreator('/ai-textbook/docs/cv/image-processing', 'b5f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/cv/introduction',
                component: ComponentCreator('/ai-textbook/docs/cv/introduction', '0f9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/cv/object-detection',
                component: ComponentCreator('/ai-textbook/docs/cv/object-detection', 'd09'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/foundations/ai-ethics',
                component: ComponentCreator('/ai-textbook/docs/foundations/ai-ethics', 'f49'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/foundations/core-concepts',
                component: ComponentCreator('/ai-textbook/docs/foundations/core-concepts', '067'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/foundations/history',
                component: ComponentCreator('/ai-textbook/docs/foundations/history', 'f68'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/foundations/types-of-ai',
                component: ComponentCreator('/ai-textbook/docs/foundations/types-of-ai', '840'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/getting-started/installation',
                component: ComponentCreator('/ai-textbook/docs/getting-started/installation', '9dd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/getting-started/quick-start',
                component: ComponentCreator('/ai-textbook/docs/getting-started/quick-start', '2e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/intro',
                component: ComponentCreator('/ai-textbook/docs/intro', '41a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/ml/deep-learning',
                component: ComponentCreator('/ai-textbook/docs/ml/deep-learning', '1ff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/ml/introduction',
                component: ComponentCreator('/ai-textbook/docs/ml/introduction', '493'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/ml/reinforcement-learning',
                component: ComponentCreator('/ai-textbook/docs/ml/reinforcement-learning', 'b93'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/ml/supervised-learning',
                component: ComponentCreator('/ai-textbook/docs/ml/supervised-learning', '618'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/ml/unsupervised-learning',
                component: ComponentCreator('/ai-textbook/docs/ml/unsupervised-learning', 'fef'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/nlp/applications',
                component: ComponentCreator('/ai-textbook/docs/nlp/applications', '777'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/nlp/introduction',
                component: ComponentCreator('/ai-textbook/docs/nlp/introduction', '3f8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/nlp/language-models',
                component: ComponentCreator('/ai-textbook/docs/nlp/language-models', '9e1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/nlp/text-processing',
                component: ComponentCreator('/ai-textbook/docs/nlp/text-processing', '658'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/ai-textbook/docs/quiz',
                component: ComponentCreator('/ai-textbook/docs/quiz', '530'),
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
    path: '/ai-textbook/',
    component: ComponentCreator('/ai-textbook/', '88e'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
