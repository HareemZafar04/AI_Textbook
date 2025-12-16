import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', '7af'),
    exact: true
  },
  {
    path: '/blog/2023/01/01/welcome',
    component: ComponentCreator('/blog/2023/01/01/welcome', 'f7f'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', '182'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '287'),
    exact: true
  },
  {
    path: '/blog/tags/ai',
    component: ComponentCreator('/blog/tags/ai', '52e'),
    exact: true
  },
  {
    path: '/blog/tags/education',
    component: ComponentCreator('/blog/tags/education', '045'),
    exact: true
  },
  {
    path: '/blog/tags/genai',
    component: ComponentCreator('/blog/tags/genai', '37f'),
    exact: true
  },
  {
    path: '/blog/tags/machine-learning',
    component: ComponentCreator('/blog/tags/machine-learning', '524'),
    exact: true
  },
  {
    path: '/blog/tags/research',
    component: ComponentCreator('/blog/tags/research', '57a'),
    exact: true
  },
  {
    path: '/blog/tags/robotics',
    component: ComponentCreator('/blog/tags/robotics', '988'),
    exact: true
  },
  {
    path: '/quiz',
    component: ComponentCreator('/quiz', 'ed5'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '74f'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '089'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'dd7'),
            routes: [
              {
                path: '/docs/applications/autonomous-systems',
                component: ComponentCreator('/docs/applications/autonomous-systems', 'abd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/finance',
                component: ComponentCreator('/docs/applications/finance', 'bb6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/healthcare',
                component: ComponentCreator('/docs/applications/healthcare', '744'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/applications/robotics',
                component: ComponentCreator('/docs/applications/robotics', '67e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/applications',
                component: ComponentCreator('/docs/cv/applications', 'ed1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/image-processing',
                component: ComponentCreator('/docs/cv/image-processing', '27c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/introduction',
                component: ComponentCreator('/docs/cv/introduction', 'c42'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/cv/object-detection',
                component: ComponentCreator('/docs/cv/object-detection', '444'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/ai-ethics',
                component: ComponentCreator('/docs/foundations/ai-ethics', '4ce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/core-concepts',
                component: ComponentCreator('/docs/foundations/core-concepts', 'b67'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/history',
                component: ComponentCreator('/docs/foundations/history', '37a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/foundations/types-of-ai',
                component: ComponentCreator('/docs/foundations/types-of-ai', '0e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '267'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/quick-start',
                component: ComponentCreator('/docs/getting-started/quick-start', '09c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/deep-learning',
                component: ComponentCreator('/docs/ml/deep-learning', 'ed0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/introduction',
                component: ComponentCreator('/docs/ml/introduction', '79f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/reinforcement-learning',
                component: ComponentCreator('/docs/ml/reinforcement-learning', '55d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/supervised-learning',
                component: ComponentCreator('/docs/ml/supervised-learning', '637'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ml/unsupervised-learning',
                component: ComponentCreator('/docs/ml/unsupervised-learning', 'b2d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/applications',
                component: ComponentCreator('/docs/nlp/applications', 'c51'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/introduction',
                component: ComponentCreator('/docs/nlp/introduction', '631'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/language-models',
                component: ComponentCreator('/docs/nlp/language-models', '47a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/nlp/text-processing',
                component: ComponentCreator('/docs/nlp/text-processing', '63e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/quiz',
                component: ComponentCreator('/docs/quiz', 'b0b'),
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
    component: ComponentCreator('/', 'fd5'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
