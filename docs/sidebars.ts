import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Main documentation sidebar based on Research Assistant project structure
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: '🚀 Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/configuration',
        'getting-started/first-query',
      ],
    },
    {
      type: 'category',
      label: '🏗️ System Architecture',
      items: [
        'architecture/overview',
        'architecture/rag-pipeline',
        'architecture/hybrid-search',
        'architecture/data-flow',
        'architecture/component-interactions',
      ],
    },
    // TODO: Add more sections as we create the documentation
    // Additional sections will be uncommented as we create the content
    {
      type: 'category',
      label: '🖥️ User Interfaces',
      items: [
        'interfaces/streamlit-app',
      ],
    },
    {
      type: 'category',
      label: '⚡ Performance & Scalability',
      items: [
        'performance/optimization',
      ],
    },
    {
      type: 'category',
      label: '🧪 Testing & Evaluation',
      items: [
        'testing/evaluation-framework',
      ],
    },
    {
      type: 'category',
      label: '📚 Reference',
      items: [
        'reference/evaluation-scripts',
        'reference/metrics',
        'reference/faq',
        'reference/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: '🔧 Troubleshooting',
      items: [
        'troubleshooting/evaluation',
      ],
    },
  ],
};

export default sidebars;
