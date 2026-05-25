import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const parts = [
  ['00', 'Vì sao cần Tensor Parallelism', 'Bộ nhớ một GPU không vừa mô hình lớn. Phân biệt DP, TP, PP, FSDP. Khi nào chọn TP.', '/docs/00-tai-sao/01-overview'],
  ['01', 'Toán học của Tensor Parallelism', 'Phép nhân ma trận shard cột và shard hàng. Derivation Y = XW. All-reduce, all-gather, reduce-scatter.', '/docs/01-toan-hoc/01-overview'],
  ['02', 'PyTorch Distributed primitives', 'ProcessGroup, NCCL, DeviceMesh và DTensor với Replicate, Shard, Partial placement.', '/docs/02-pytorch-distributed/01-overview'],
  ['03', 'Tensor Parallelism cho MLP', 'Megatron Col-then-Row pattern. SwiGLU với ba linear w1, w2, w3. Walkthrough toy simple model.', '/docs/03-tp-cho-mlp/01-overview'],
  ['04', 'Tensor Parallelism cho Self-Attention', 'Shard theo head, không theo head_dim. Pattern Col-Col-Col-Row cho QKVO. GQA và MQA với TP.', '/docs/04-tp-cho-attention/01-overview'],
  ['05', 'Embedding và LM Head', 'Embedding shard vocab. LM Head với loss_parallel để tính cross-entropy không all-gather logits.', '/docs/05-embedding-lm-head/01-overview'],
  ['06', 'Sequence Parallelism', 'Shard chiều sequence ở zone norm và residual. AllGather + ReduceScatter thay AllReduce.', '/docs/06-sequence-parallel/01-overview'],
  ['07', '2D Parallelism với FSDP', 'Kết hợp TP intra-layer với FSDP shard parameter qua DP mesh. Mixed precision và activation checkpoint.', '/docs/07-2d-parallelism-fsdp/01-overview'],
  ['08', 'Walkthrough Llama-3 toy', 'Đọc đầy đủ model.py, parallelism.py, train.py của toy LLM. End-to-end một step với placement chính xác.', '/docs/08-llama3-walkthrough/01-overview'],
  ['09', 'Performance và Debugging', 'Overlap collective với compute, profiling, MFU, và checklist debug các bug TP/FSDP thường gặp.', '/docs/09-performance-debugging/01-overview'],
];

function HomepageHeader(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <Heading as="h1" className={styles.heroTitle}>{siteConfig.title}</Heading>
        <p className={styles.heroTagline}>{siteConfig.tagline}</p>
        <div className={styles.heroButtons}>
          <Link className={`button button--primary button--lg ${styles.heroButton}`} to="/docs/intro">Bắt đầu đọc</Link>
          <Link className={`button button--secondary button--lg ${styles.heroButton}`} to="/docs/00-tai-sao/01-overview">Vào Phần 0</Link>
        </div>
      </div>
    </header>
  );
}

function PartGrid(): ReactNode {
  return (
    <section className={styles.gridSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>Chuỗi bài giảng đi sâu</Heading>
        <p className={styles.sectionSubtitle}>
          Đi từ trực giác về bộ nhớ GPU, tới toán học của ma trận shard, tới primitives của PyTorch Distributed, rồi áp dụng vào MLP, Transformer và toy code Llama-3 chạy được.
        </p>
        <div className={styles.grid}>
          {parts.map(([number, title, description, to]) => (
            <Link key={number} to={to} className={styles.card}>
              <div className={styles.cardNumber}>PHẦN {number}</div>
              <Heading as="h3" className={styles.cardTitle}>{title}</Heading>
              <p className={styles.cardDescription}>{description}</p>
              <span className={styles.badgeReady}>Đọc phần này</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function PhilosophySection(): ReactNode {
  return (
    <section className={styles.philosophy}>
      <div className="container">
        <blockquote className={styles.quote}>
          <p><em>Tensor Parallelism không phải mẹo kỹ thuật. Nó là kết quả tự nhiên khi ta đặt câu hỏi: nếu ma trận trọng số quá lớn so với một GPU, hệ thống nên sharding như thế nào để vẫn đúng về toán, hiệu quả về băng thông, và sạch về cấu trúc code.</em></p>
        </blockquote>
        <p className={styles.philosophyText}>
          Mục tiêu của chuỗi này là giúp bạn hiểu tường tận: từ phép nhân ma trận shard, qua các collective operation, lên tới Megatron MLP pattern, attention pattern, và cuối cùng là 2D parallelism với FSDP. Mỗi khái niệm có ba lớp giải thích: trực giác hình học, công thức toán, và code PyTorch thật chạy được trên repo toy đi kèm.
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline as string}>
      <HomepageHeader />
      <main>
        <PartGrid />
        <PhilosophySection />
      </main>
    </Layout>
  );
}
