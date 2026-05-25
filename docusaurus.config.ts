import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Tensor Parallelism Deep Dive',
  tagline: 'Bài giảng đi sâu vào bản chất Tensor Parallelism: toán học, PyTorch primitives, MLP và Transformer, 2D parallelism với FSDP, và walkthrough toy code Llama-3',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
    faster: true,
  },

  url: 'https://tuandung222.github.io',
  baseUrl: '/Tensor_Parallelism_Toy_Example/',
  organizationName: 'tuandung222',
  projectName: 'Tensor_Parallelism_Toy_Example',
  trailingSlash: false,
  onBrokenLinks: 'warn',

  headTags: [
    {
      tagName: 'meta',
      attributes: {
        name: 'robots',
        content: 'noindex,nofollow,noarchive,nosnippet',
      },
    },
  ],

  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
    localeConfigs: {
      vi: {label: 'Tiếng Việt', htmlLang: 'vi-VN'},
    },
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl: 'https://github.com/tuandung222/Tensor_Parallelism_Toy_Example/edit/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: false,
          numberPrefixParser: false,
        },
        blog: false,
        sitemap: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en', 'vi'],
        indexBlog: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
    navbar: {
      title: 'TP Deep Dive',
      logo: {
        alt: 'Tensor Parallelism Deep Dive',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'lectureSidebar',
          position: 'left',
          label: 'Bài giảng',
        },
        {
          to: '/docs/resources/glossary',
          label: 'Thuật ngữ',
          position: 'left',
        },
        {
          href: 'https://github.com/tuandung222/Tensor_Parallelism_Toy_Example',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Trục nội dung',
          items: [
            {label: 'Vì sao cần TP', to: '/docs/00-tai-sao/01-overview'},
            {label: 'Toán học TP', to: '/docs/01-toan-hoc/01-overview'},
            {label: 'PyTorch primitives', to: '/docs/02-pytorch-distributed/01-overview'},
            {label: 'TP cho MLP', to: '/docs/03-tp-cho-mlp/01-overview'},
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {label: 'Glossary', to: '/docs/resources/glossary'},
            {label: 'Cheatsheet', to: '/docs/resources/cheatsheet'},
            {label: 'References', to: '/docs/resources/references'},
          ],
        },
      ],
      copyright: `Bản quyền © ${new Date().getFullYear()} Tensor Parallelism Deep Dive. Nội dung đang được biên soạn.`,
    },
    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['bash', 'json', 'typescript', 'yaml', 'python', 'markdown', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
