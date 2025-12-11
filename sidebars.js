/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
module.exports = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/installation', 'getting-started/quick-start'],
    },
    {
      type: 'category',
      label: 'Foundations of AI',
      items: [
        'foundations/history',
        'foundations/core-concepts',
        'foundations/types-of-ai',
        'foundations/ai-ethics',
      ],
    },
    {
      type: 'category',
      label: 'Machine Learning',
      items: [
        'ml/introduction',
        'ml/supervised-learning',
        'ml/unsupervised-learning',
        'ml/reinforcement-learning',
        'ml/deep-learning',
      ],
    },
    {
      type: 'category',
      label: 'Natural Language Processing',
      items: [
        'nlp/introduction',
        'nlp/text-processing',
        'nlp/language-models',
        'nlp/applications',
      ],
    },
    {
      type: 'category',
      label: 'Computer Vision',
      items: [
        'cv/introduction',
        'cv/image-processing',
        'cv/object-detection',
        'cv/applications',
      ],
    },
    {
      type: 'category',
      label: 'AI Applications',
      items: [
        'applications/healthcare',
        'applications/finance',
        'applications/autonomous-systems',
        'applications/robotics',
      ],
    },
    // {
    //   type: 'category',
    //   label: 'Future of AI',
    //   items: [
    //     'future/advancements',
    //     'future/implications',
    //     'future/agi',
    //   ],
    // },
    'quiz',
  ],
};