import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  lectureSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Giới thiệu chuỗi bài giảng',
    },
    {
      type: 'category',
      label: 'Phần 0: Vì sao cần Tensor Parallelism',
      link: {type: 'doc', id: '00-tai-sao/01-overview'},
      collapsed: false,
      items: [
        '00-tai-sao/01-overview',
        '00-tai-sao/02-mo-hinh-lon-va-bo-nho-gpu',
        '00-tai-sao/03-bon-loai-parallelism',
        '00-tai-sao/04-khi-nao-chon-tp',
      ],
    },
    {
      type: 'category',
      label: 'Phần 1: Toán học của Tensor Parallelism',
      link: {type: 'doc', id: '01-toan-hoc/01-overview'},
      collapsed: false,
      items: [
        '01-toan-hoc/01-overview',
        '01-toan-hoc/02-on-lai-phep-nhan-ma-tran',
        '01-toan-hoc/03-column-parallel-derivation',
        '01-toan-hoc/04-row-parallel-derivation',
        '01-toan-hoc/05-collective-operations',
      ],
    },
    {
      type: 'category',
      label: 'Phần 2: PyTorch Distributed Primitives',
      link: {type: 'doc', id: '02-pytorch-distributed/01-overview'},
      collapsed: false,
      items: [
        '02-pytorch-distributed/01-overview',
        '02-pytorch-distributed/02-process-group-va-nccl',
        '02-pytorch-distributed/03-device-mesh',
        '02-pytorch-distributed/04-dtensor-va-placement',
      ],
    },
    {
      type: 'category',
      label: 'Phần 3: Tensor Parallelism cho MLP',
      link: {type: 'doc', id: '03-tp-cho-mlp/01-overview'},
      collapsed: false,
      items: [
        '03-tp-cho-mlp/01-overview',
        '03-tp-cho-mlp/02-megatron-mlp-pattern',
        '03-tp-cho-mlp/03-swiglu-ba-linear',
        '03-tp-cho-mlp/04-toy-simple-model-walkthrough',
      ],
    },
    {
      type: 'category',
      label: 'Tài nguyên',
      collapsed: true,
      items: [
        'resources/glossary',
        'resources/cheatsheet',
        'resources/references',
      ],
    },
  ],
};

export default sidebars;
