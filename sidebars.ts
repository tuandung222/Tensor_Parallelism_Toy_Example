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
      label: 'Phần 4: Tensor Parallelism cho Self-Attention',
      link: {type: 'doc', id: '04-tp-cho-attention/01-overview'},
      collapsed: false,
      items: [
        '04-tp-cho-attention/01-overview',
        '04-tp-cho-attention/02-multi-head-shard',
        '04-tp-cho-attention/03-qkv-output-pattern',
        '04-tp-cho-attention/04-gqa-mqa-kv-heads',
        '04-tp-cho-attention/05-attention-walkthrough',
      ],
    },
    {
      type: 'category',
      label: 'Phần 5: Embedding và LM Head',
      link: {type: 'doc', id: '05-embedding-lm-head/01-overview'},
      collapsed: false,
      items: [
        '05-embedding-lm-head/01-overview',
        '05-embedding-lm-head/02-embedding-rowwise',
        '05-embedding-lm-head/03-output-colwise-loss-parallel',
        '05-embedding-lm-head/04-weight-tying-considerations',
        '05-embedding-lm-head/05-embedding-lm-head-walkthrough',
      ],
    },
    {
      type: 'category',
      label: 'Phần 6: Sequence Parallelism',
      link: {type: 'doc', id: '06-sequence-parallel/01-overview'},
      collapsed: false,
      items: [
        '06-sequence-parallel/01-overview',
        '06-sequence-parallel/02-motivation-activation-memory',
        '06-sequence-parallel/03-norm-sequence-parallel',
        '06-sequence-parallel/04-prepare-module-input-transitions',
        '06-sequence-parallel/05-sp-cost-analysis',
      ],
    },
    {
      type: 'category',
      label: 'Phần 7: 2D Parallelism với FSDP',
      link: {type: 'doc', id: '07-2d-parallelism-fsdp/01-overview'},
      collapsed: false,
      items: [
        '07-2d-parallelism-fsdp/01-overview',
        '07-2d-parallelism-fsdp/02-fsdp-co-ban',
        '07-2d-parallelism-fsdp/03-mesh-2d-va-luong-gradient',
        '07-2d-parallelism-fsdp/04-mixed-precision-va-checkpoint',
        '07-2d-parallelism-fsdp/05-fsdp-walkthrough',
      ],
    },
    {
      type: 'category',
      label: 'Phần 8: Walkthrough Llama-3 toy',
      link: {type: 'doc', id: '08-llama3-walkthrough/01-overview'},
      collapsed: false,
      items: [
        '08-llama3-walkthrough/01-overview',
        '08-llama3-walkthrough/02-model-py',
        '08-llama3-walkthrough/03-parallelism-py',
        '08-llama3-walkthrough/04-train-py',
        '08-llama3-walkthrough/05-end-to-end-walk',
      ],
    },
    {
      type: 'category',
      label: 'Phần 9: Performance và Debugging',
      link: {type: 'doc', id: '09-performance-debugging/01-overview'},
      collapsed: false,
      items: [
        '09-performance-debugging/01-overview',
        '09-performance-debugging/02-overlap-collective-va-compute',
        '09-performance-debugging/03-activation-checkpoint-va-bo-nho',
        '09-performance-debugging/04-profiling-va-mfu',
        '09-performance-debugging/05-debug-common-pitfalls',
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
