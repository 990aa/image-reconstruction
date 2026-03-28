import Reveal from 'reveal.js';
import 'reveal.js/dist/reveal.css';
import 'reveal.js/dist/theme/white.css';
import './style.css';

type BestRunStats = {
  target: string;
  total_iterations: number;
  accepted_polygons: number;
  final_mse: number;
  runtime_seconds: number;
};

type RunStatsPayload = {
  best_run: BestRunStats;
};

type Slide = {
  id: string;
  content: string;
};

const DEFAULT_STATS: BestRunStats = {
  target: 'heart',
  total_iterations: 5000,
  accepted_polygons: 0,
  final_mse: 0,
  runtime_seconds: 0,
};

async function loadRunStats(): Promise<BestRunStats> {
  try {
    const response = await fetch('/images/run_stats.json', { cache: 'no-store' });
    if (!response.ok) {
      return DEFAULT_STATS;
    }
    const payload = (await response.json()) as RunStatsPayload;
    return payload.best_run ?? DEFAULT_STATS;
  } catch {
    return DEFAULT_STATS;
  }
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function createSlides(stats: BestRunStats): Slide[] {
  return [
    {
      id: 'title',
      content: `
        <div class="title-shell minimal">
          <h1>Attention-Guided Evolutionary Art</h1>
          <p class="subtitle">Teaching an AI to paint using math.</p>
          <p class="author">Abdul Ahad</p>
        </div>
      `,
    },
    {
      id: 'what-does-it-do',
      content: `
        <div class="split two-col">
          <div class="copy">
            <h2>What does it do?</h2>
            <p class="lead">An AI builds an image from scratch using only triangles, quadrilaterals, and ellipses, placed one at a time.</p>
            <ul>
              <li>Starts with a blank white canvas.</li>
              <li>Adds shapes one at a time.</li>
              <li>Each shape makes the image more accurate.</li>
            </ul>
          </div>
          <div class="media-panel">
            <img src="/images/heart_final.jpg" alt="Final heart canvas" />
          </div>
        </div>
      `,
    },
    {
      id: 'error-map',
      content: `
        <div class="split two-col">
          <div class="media-panel">
            <img src="/images/heart_error_map_mid.jpg" alt="Mid-run error map" />
          </div>
          <div class="copy">
            <h2>The error map</h2>
            <p class="lead">Bright areas = where the AI is doing worst. Every new shape spawns inside a bright zone.</p>
            <p>We measure pixel-by-pixel color difference to find the worst regions.</p>
          </div>
        </div>
      `,
    },
    {
      id: 'mse-math',
      content: `
        <div class="formula-slide">
          <h2>The math: MSE</h2>
          <img class="formula" src="/images/mse_formula.png" alt="MSE formula" />
          <p>We compute this for every pixel.</p>
          <p>Bright on the error map = high MSE at that pixel.</p>
          <p>MSE drops as the image gets better.</p>
        </div>
      `,
    },
    {
      id: 'hill-climbing',
      content: `
        <div class="hill-slide">
          <h2>Accept or reject?</h2>
          <svg viewBox="0 0 1040 440" class="decision-diagram" role="img" aria-label="Hill climbing decision flow">
            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#2d3a5c"></path>
              </marker>
            </defs>
            <rect x="40" y="120" width="250" height="120" rx="18" class="node" />
            <text x="165" y="183" text-anchor="middle">Try a shape</text>

            <polygon points="460,80 610,180 460,280 310,180" class="decision" />
            <text x="460" y="172" text-anchor="middle">Did MSE</text>
            <text x="460" y="205" text-anchor="middle">improve?</text>

            <rect x="730" y="40" width="250" height="120" rx="18" class="node yes" />
            <text x="855" y="103" text-anchor="middle">Yes: keep it</text>

            <rect x="730" y="280" width="250" height="120" rx="18" class="node no" />
            <text x="855" y="343" text-anchor="middle">No: throw it away</text>

            <line x1="290" y1="180" x2="310" y2="180" class="arrow" marker-end="url(#arrow)" />
            <line x1="610" y1="130" x2="730" y2="100" class="arrow" marker-end="url(#arrow)" />
            <line x1="610" y1="230" x2="730" y2="320" class="arrow" marker-end="url(#arrow)" />

            <text x="655" y="112" class="label">Yes</text>
            <text x="655" y="300" class="label">No</text>
          </svg>
          <p class="explain">The AI tries thousands of shapes per second. Most are rejected. The ones that survive are exactly the ones that make the image better.</p>
        </div>
      `,
    },
    {
      id: 'three-phases',
      content: `
        <div class="phase-slide">
          <h2>How the AI evolves</h2>
          <div class="phase-grid">
            <article class="phase-card">
              <h3>Phase 1 · Coarse</h3>
              <img src="/images/heart_phase_coarse.jpg" alt="Coarse phase frame" />
              <p>Large polygons quickly carve out dominant color regions and global silhouette.</p>
            </article>
            <article class="phase-card">
              <h3>Phase 2 · Structural</h3>
              <img src="/images/heart_phase_structural.jpg" alt="Structural phase frame" />
              <p>Medium polygons lock edges and boundaries into stable structure.</p>
            </article>
            <article class="phase-card">
              <h3>Phase 3 · Detail</h3>
              <img src="/images/heart_phase_detail.jpg" alt="Detail phase frame" />
              <p>Small polygons refine local errors and sharpen final appearance.</p>
            </article>
          </div>
        </div>
      `,
    },
    {
      id: 'live-demo-still',
      content: `
        <div class="full-bleed">
          <img src="/images/heart_live_panel.jpg" alt="Four-panel live demo still" />
          <p class="caption">This is the AI thinking in real time.</p>
        </div>
      `,
    },
    {
      id: 'three-targets',
      content: `
        <div class="comparison-slide">
          <h2>Three targets, three results</h2>
          <img src="/images/comparison_grid.jpg" alt="Three-by-three comparison grid" />
          <p class="caption">The same algorithm produces a completely unique arrangement of shapes every run.</p>
        </div>
      `,
    },
    {
      id: 'why-matters',
      content: `
        <div class="copy narrow">
          <h2>Why this matters</h2>
          <ul>
            <li>Error map = attention. The model focuses computation where mistakes are largest.</li>
            <li>Polygon size schedule = learning rate decay. Big early moves, fine late tuning.</li>
            <li>Accept/reject = gradient descent without calculus. Keep only error-reducing updates.</li>
          </ul>
        </div>
      `,
    },
    {
      id: 'summary',
      content: `
        <div class="summary-slide">
          <h2>Summary</h2>
          <div class="stats-grid">
            <div class="stat">
              <span class="label">Total Iterations</span>
              <span class="value">${formatNumber(stats.total_iterations)}</span>
            </div>
            <div class="stat">
              <span class="label">Polygons Accepted</span>
              <span class="value">${formatNumber(stats.accepted_polygons)}</span>
            </div>
            <div class="stat">
              <span class="label">Final MSE</span>
              <span class="value">${stats.final_mse.toFixed(6)}</span>
            </div>
            <div class="stat">
              <span class="label">Runtime</span>
              <span class="value">${stats.runtime_seconds.toFixed(2)} s</span>
            </div>
          </div>
          <p class="closing">Every run is unique. Every shape was chosen by math.</p>
        </div>
      `,
    },
  ];
}

async function bootstrap(): Promise<void> {
  const app = document.querySelector<HTMLDivElement>('#app');
  if (!app) {
    throw new Error('Missing #app container');
  }

  const stats = await loadRunStats();
  const slides = createSlides(stats);

  app.innerHTML = '<div class="reveal"><div class="slides"></div></div>';
  const slidesRoot = app.querySelector<HTMLDivElement>('.slides');
  if (!slidesRoot) {
    throw new Error('Missing .slides container');
  }

  for (const slide of slides) {
    const section = document.createElement('section');
    section.className = `slide-${slide.id}`;
    section.innerHTML = slide.content;
    slidesRoot.appendChild(section);
  }

  const revealRoot = app.querySelector<HTMLElement>('.reveal');
  if (!revealRoot) {
    throw new Error('Missing .reveal container');
  }

  const deck = new Reveal(revealRoot, {
    controls: true,
    progress: true,
    hash: true,
    center: true,
  });

  await deck.initialize();
}

bootstrap().catch((error: unknown) => {
  console.error('Failed to initialize slide deck:', error);
});
