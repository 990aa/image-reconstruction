import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PptxGenJS from 'pptxgenjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const slidesRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(slidesRoot, '..');
const outputsDir = path.resolve(repoRoot, 'python', 'outputs');
const docsFigDir = path.resolve(repoRoot, 'docs', 'figures');
const distDir = path.resolve(slidesRoot, 'dist');

const pptFileName = 'Attention_Guided_Evolutionary_Art_Abdul_Ahad.pptx';
const pptDistPath = path.resolve(distDir, pptFileName);
const pptRootPath = path.resolve(repoRoot, pptFileName);

const NAME = 'Abdul Ahad';
const REG_NO = '245805010';

const COLORS = {
  bg: 'F8FAFC',
  panel: 'FFFFFF',
  panelSoft: 'FFF7ED',
  ink: '0F172A',
  muted: '475569',
  line: 'CBD5E1',
  accent: 'EA580C',
  accentSoft: 'FFEDD5',
  greenSoft: 'DCFCE7',
  blueSoft: 'E0F2FE',
  violetSoft: 'EDE9FE',
};

const FONT = {
  heading: 'Aptos Display',
  body: 'Aptos',
  mono: 'Consolas',
};

function requiredFile(baseDir, fileName) {
  const p = path.resolve(baseDir, fileName);
  if (!fs.existsSync(p)) {
    throw new Error(`Missing required asset: ${p}`);
  }
  return p;
}

function outputImg(fileName) {
  return requiredFile(outputsDir, fileName);
}

function figureImg(fileName) {
  return requiredFile(docsFigDir, fileName);
}

function loadStats() {
  const statsPath = path.resolve(outputsDir, 'run_stats.json');
  if (!fs.existsSync(statsPath)) {
    return {
      best_run: {
        target: 'heart',
        total_iterations: 5000,
        accepted_polygons: 0,
        final_mse: 0,
        runtime_seconds: 0,
      },
    };
  }
  return JSON.parse(fs.readFileSync(statsPath, 'utf8'));
}

function addFrame(slide, title, subtitle = '') {
  slide.background = { color: COLORS.bg };

  slide.addShape('rect', {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.12,
    fill: { color: COLORS.accent },
    line: { color: COLORS.accent },
  });

  slide.addText(title, {
    x: 0.55,
    y: 0.2,
    w: 11.8,
    h: 0.45,
    fontFace: FONT.heading,
    fontSize: 24,
    bold: true,
    color: COLORS.ink,
  });

  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.58,
      y: 0.63,
      w: 11.8,
      h: 0.25,
      fontFace: FONT.body,
      fontSize: 12,
      color: COLORS.muted,
    });
  }

  slide.addShape('rect', {
    x: 0,
    y: 7.16,
    w: 13.333,
    h: 0.34,
    fill: { color: 'FFFFFF' },
    line: { color: COLORS.line, pt: 0.5 },
  });

  slide.addText(`Name: ${NAME} | Reg. No.: ${REG_NO}`, {
    x: 0.55,
    y: 7.22,
    w: 8.3,
    h: 0.2,
    fontFace: FONT.body,
    fontSize: 10,
    color: COLORS.muted,
  });

  slide.addText('Home', {
    x: 12.15,
    y: 7.21,
    w: 0.7,
    h: 0.2,
    align: 'center',
    fontFace: FONT.body,
    fontSize: 9,
    color: '1D4ED8',
    hyperlink: { slide: 1 },
  });
}

function addCard(slide, x, y, w, h, fill = COLORS.panel) {
  slide.addShape('roundRect', {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: COLORS.line, pt: 0.8 },
    shadow: {
      type: 'outer',
      color: '94A3B8',
      blur: 2,
      angle: 45,
      distance: 1,
      opacity: 0.2,
    },
  });
}

function addBullets(slide, lines, { x, y, w, h, fontSize = 15 }) {
  slide.addText(
    lines.map((line) => ({ text: line, options: { bullet: { indent: 14 } } })),
    {
      x,
      y,
      w,
      h,
      fontFace: FONT.body,
      fontSize,
      color: COLORS.ink,
      breakLine: true,
      paraSpaceAfterPt: 8,
      valign: 'top',
    }
  );
}

function addDeck(ppt, stats) {
  const best = stats.best_run;

  // Slide 1: Title
  const s1 = ppt.addSlide();
  s1.background = { color: 'FFF7ED' };
  addCard(s1, 0.6, 1.0, 12.1, 5.6, 'FFFFFF');
  s1.addText('Attention-Guided Evolutionary Art', {
    x: 1.0,
    y: 2.0,
    w: 10.8,
    h: 0.9,
    fontFace: FONT.heading,
    fontSize: 40,
    bold: true,
    color: COLORS.ink,
  });
  s1.addText('Teaching an AI to paint with optimization and error feedback', {
    x: 1.0,
    y: 3.0,
    w: 10.3,
    h: 0.45,
    fontFace: FONT.body,
    fontSize: 19,
    color: COLORS.muted,
  });
  s1.addText(`Name: ${NAME}\nReg. No.: ${REG_NO}`, {
    x: 1.0,
    y: 4.1,
    w: 4.5,
    h: 0.9,
    breakLine: true,
    fontFace: FONT.body,
    fontSize: 16,
    color: COLORS.ink,
  });
  s1.addShape('roundRect', {
    x: 9.75,
    y: 5.45,
    w: 2.3,
    h: 0.62,
    fill: { color: COLORS.accent },
    line: { color: COLORS.accent },
  });
  s1.addText('Start', {
    x: 9.75,
    y: 5.59,
    w: 2.3,
    h: 0.25,
    align: 'center',
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: 'FFFFFF',
    hyperlink: { slide: 2 },
  });

  // Slide 2: Intuition and goal
  const s2 = ppt.addSlide();
  addFrame(s2, 'What this AI is trying to achieve', 'Goal state and simple intuition');
  addCard(s2, 0.6, 1.05, 6.25, 5.95, COLORS.panel);
  s2.addText('Goal state', {
    x: 0.95,
    y: 1.35,
    w: 2.2,
    h: 0.3,
    fontFace: FONT.body,
    fontSize: 16,
    bold: true,
    color: COLORS.ink,
  });
  s2.addText('Produce a canvas that is visually indistinguishable from the target while minimizing MSE.', {
    x: 0.95,
    y: 1.72,
    w: 5.5,
    h: 0.9,
    fontFace: FONT.body,
    fontSize: 14,
    color: COLORS.muted,
  });
  addBullets(
    s2,
    [
      'Start with a fully white canvas.',
      'Propose one geometric shape at a time.',
      'Measure if the proposal lowers global error.',
      'Keep only proposals that improve the objective.',
      'Repeat until the canvas converges to the target.',
    ],
    { x: 0.92, y: 2.78, w: 5.7, h: 3.2, fontSize: 14 }
  );
  s2.addImage({ path: outputImg('heart_final.jpg'), x: 7.1, y: 1.05, w: 5.65, h: 5.95 });

  // Slide 3: System architecture (report-quality diagram)
  const s3 = ppt.addSlide();
  addFrame(s3, 'System architecture of the model', 'How data flows through the algorithm');
  s3.addImage({ path: figureImg('architecture_diagram.png'), x: 0.72, y: 1.15, w: 12.0, h: 5.9 });

  // Slide 4: Error map + per-pixel formula and terms
  const s4 = ppt.addSlide();
  addFrame(s4, 'Attention mechanism: Per-pixel error map', 'Where the AI chooses to spend effort');
  addCard(s4, 0.62, 1.1, 5.75, 5.8, COLORS.blueSoft);
  s4.addImage({ path: outputImg('heart_error_map_mid.jpg'), x: 0.88, y: 1.45, w: 5.2, h: 3.45 });
  s4.addText('Per-pixel error formula', {
    x: 0.9,
    y: 5.05,
    w: 4.9,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 12,
    bold: true,
    color: COLORS.ink,
  });
  s4.addText('E(y,x) = (R_t - R_c)^2 + (G_t - G_c)^2 + (B_t - B_c)^2', {
    x: 0.9,
    y: 5.3,
    w: 5.3,
    h: 0.45,
    fontFace: FONT.mono,
    fontSize: 11,
    color: COLORS.ink,
  });

  addCard(s4, 6.6, 1.1, 6.15, 5.8, COLORS.panel);
  s4.addText('Meaning of terms', {
    x: 6.9,
    y: 1.38,
    w: 4.0,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 15,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s4,
    [
      'y,x: pixel coordinates on the image grid.',
      'R_t,G_t,B_t: target RGB values at that pixel.',
      'R_c,G_c,B_c: current canvas RGB values.',
      'E(y,x): local mismatch energy at that pixel.',
      'High E(y,x) means the model is still wrong there.',
      'Sampling is biased toward high-E regions, creating attention.',
    ],
    { x: 6.82, y: 1.78, w: 5.75, h: 3.55, fontSize: 13 }
  );
  s4.addText('Why this helps goal-state convergence:', {
    x: 6.9,
    y: 5.45,
    w: 5.6,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 12,
    bold: true,
    color: COLORS.ink,
  });
  s4.addText('By placing proposals where errors are largest, each accepted shape has higher expected impact on global objective reduction.', {
    x: 6.9,
    y: 5.67,
    w: 5.6,
    h: 0.9,
    fontFace: FONT.body,
    fontSize: 12,
    color: COLORS.muted,
  });

  // Slide 5: Global MSE formula and objective
  const s5 = ppt.addSlide();
  addFrame(s5, 'Objective function: Mean Squared Error (MSE)', 'How progress is measured globally');
  addCard(s5, 0.62, 1.15, 12.1, 5.8, COLORS.panelSoft);
  s5.addImage({ path: outputImg('mse_formula.png'), x: 1.05, y: 1.55, w: 11.2, h: 1.3 });

  s5.addText('Term-by-term interpretation', {
    x: 1.0,
    y: 3.1,
    w: 4.2,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s5,
    [
      'n: total scalar values (H x W x 3 channels).',
      'Y_i: target value for scalar i.',
      'Y_hat_i: current canvas value for scalar i.',
      '(Y_i - Y_hat_i)^2: squared reconstruction residual.',
    ],
    { x: 0.98, y: 3.45, w: 5.6, h: 2.0, fontSize: 13 }
  );

  s5.addText('How MSE drives the AI to the goal state', {
    x: 6.6,
    y: 3.1,
    w: 5.4,
    h: 0.28,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s5,
    [
      'Every candidate shape is evaluated by its new global MSE.',
      'If new MSE < current MSE, candidate is accepted.',
      'Otherwise, candidate is rejected and state is unchanged.',
      'Repeated strict improvement steps push the canvas toward the target.',
    ],
    { x: 6.58, y: 3.45, w: 5.9, h: 2.0, fontSize: 13 }
  );

  // Slide 6: Step-by-step optimization flowchart
  const s6 = ppt.addSlide();
  addFrame(s6, 'Step-by-step optimization flow', 'What happens inside the AI at every iteration');
  s6.addImage({ path: figureImg('optimization_flow.png'), x: 0.75, y: 1.1, w: 11.95, h: 4.55 });
  s6.addText('Step 1: Build current error map from target vs. canvas.', {
    x: 0.9,
    y: 5.78,
    w: 6.0,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLORS.ink,
  });
  s6.addText('Step 2: Sample proposal center from error-based probability.', {
    x: 0.9,
    y: 6.02,
    w: 6.0,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLORS.ink,
  });
  s6.addText('Step 3: Generate shape, render candidate, compute new MSE.', {
    x: 6.7,
    y: 5.78,
    w: 5.8,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLORS.ink,
  });
  s6.addText('Step 4: Accept only if MSE decreases; repeat until convergence.', {
    x: 6.7,
    y: 6.02,
    w: 5.8,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 11,
    color: COLORS.ink,
  });

  // Slide 7: Why convergence is stable
  const s7 = ppt.addSlide();
  addFrame(s7, 'Why convergence is stable and efficient', 'Scheduling + diversity + strict acceptance');
  addCard(s7, 0.68, 1.15, 4.0, 5.8, COLORS.violetSoft);
  s7.addText('Phase scheduling', {
    x: 0.98,
    y: 1.45,
    w: 3.4,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s7,
    [
      '0–30%: size 30 -> 15 px (coarse).',
      '30–70%: size 15 -> 8 px (structure).',
      '70–100%: size 8 -> 3 px (detail).',
    ],
    { x: 0.95, y: 1.82, w: 3.5, h: 1.95, fontSize: 12 }
  );

  addCard(s7, 4.95, 1.15, 3.95, 5.8, COLORS.greenSoft);
  s7.addText('Shape diversity', {
    x: 5.25,
    y: 1.45,
    w: 3.3,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s7,
    [
      'Cycle: Triangle -> Quad -> Ellipse.',
      'Deterministic order avoids mode collapse.',
      'Different shapes model different edges and regions.',
    ],
    { x: 5.2, y: 1.82, w: 3.45, h: 1.95, fontSize: 12 }
  );

  addCard(s7, 9.2, 1.15, 3.45, 5.8, COLORS.blueSoft);
  s7.addText('Guarantee', {
    x: 9.45,
    y: 1.45,
    w: 2.9,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s7,
    [
      'Accept only strict MSE improvements.',
      'Therefore MSE is monotonic non-increasing.',
      'Final state is a local optimum under proposal model.',
    ],
    { x: 9.42, y: 1.82, w: 2.95, h: 2.2, fontSize: 12 }
  );

  s7.addImage({ path: outputImg('heart_phase_coarse.jpg'), x: 0.95, y: 4.15, w: 2.95, h: 2.35 });
  s7.addImage({ path: outputImg('heart_phase_structural.jpg'), x: 5.2, y: 4.15, w: 2.95, h: 2.35 });
  s7.addImage({ path: outputImg('heart_phase_detail.jpg'), x: 9.42, y: 4.15, w: 2.95, h: 2.35 });

  // Slide 8: Real-time visualization interpretation
  const s8 = ppt.addSlide();
  addFrame(s8, 'Live optimization view', 'How to read the four-panel dashboard');
  s8.addImage({ path: outputImg('heart_live_panel.jpg'), x: 0.72, y: 1.05, w: 8.2, h: 6.0 });

  addCard(s8, 9.1, 1.05, 3.55, 6.0, COLORS.panel);
  s8.addText('Panel guide', {
    x: 9.35,
    y: 1.3,
    w: 3.0,
    h: 0.25,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s8,
    [
      'Target: fixed objective image.',
      'Error map: where model is currently wrong.',
      'Canvas: evolving reconstruction.',
      'Stats: iteration, MSE, acceptance, phase.',
      'Bottom curve: raw + smoothed MSE decay.',
      'Colored canvas border: red -> green with convergence.',
    ],
    { x: 9.28, y: 1.7, w: 3.1, h: 4.1, fontSize: 11 }
  );
  s8.addText('Interpretation: if error map darkens while MSE curve trends down, the model is moving toward goal state.', {
    x: 9.28,
    y: 5.95,
    w: 3.15,
    h: 0.85,
    fontFace: FONT.body,
    fontSize: 10,
    color: COLORS.muted,
  });

  // Slide 9: Results
  const s9 = ppt.addSlide();
  addFrame(s9, 'Three targets, three converged results', 'Same AI, different targets, different accepted-shape histories');
  s9.addImage({ path: outputImg('comparison_grid.jpg'), x: 0.8, y: 1.05, w: 12.0, h: 5.9 });

  // Slide 10: Summary and final metrics
  const s10 = ppt.addSlide();
  addFrame(s10, 'Summary', 'Final performance and key takeaways');

  addCard(s10, 0.75, 1.2, 12.0, 4.95, COLORS.panel);

  const rows = [
    ['Best target', String(best.target)],
    ['Total iterations', String(best.total_iterations)],
    ['Accepted polygons', String(best.accepted_polygons)],
    ['Final MSE', Number(best.final_mse).toFixed(6)],
    ['Runtime (s)', Number(best.runtime_seconds).toFixed(2)],
  ];

  let rowY = 1.6;
  for (const [label, value] of rows) {
    s10.addText(label, {
      x: 1.2,
      y: rowY,
      w: 3.6,
      h: 0.35,
      fontFace: FONT.body,
      fontSize: 15,
      bold: true,
      color: COLORS.muted,
    });
    s10.addText(value, {
      x: 4.7,
      y: rowY,
      w: 3.7,
      h: 0.35,
      fontFace: FONT.mono,
      fontSize: 17,
      color: COLORS.ink,
    });
    rowY += 0.68;
  }

  s10.addText('Theory recap', {
    x: 8.25,
    y: 1.55,
    w: 3.9,
    h: 0.3,
    fontFace: FONT.body,
    fontSize: 14,
    bold: true,
    color: COLORS.ink,
  });
  addBullets(
    s10,
    [
      'Error map supplies spatial attention.',
      'MSE defines the global optimization objective.',
      'Accept/reject hill climbing enforces objective descent.',
      'Size scheduling transitions from coarse structure to detail.',
      'Final state is reached when further proposals rarely improve MSE.',
    ],
    { x: 8.2, y: 1.95, w: 4.2, h: 2.8, fontSize: 11 }
  );

  s10.addShape('roundRect', {
    x: 1.05,
    y: 6.35,
    w: 11.4,
    h: 0.68,
    fill: { color: COLORS.accentSoft },
    line: { color: COLORS.accent, pt: 1 },
  });
  s10.addText('Every accepted shape is a measurable step toward the target goal state.', {
    x: 1.2,
    y: 6.56,
    w: 11.0,
    h: 0.25,
    align: 'center',
    fontFace: FONT.body,
    fontSize: 16,
    bold: true,
    color: COLORS.ink,
  });
}

async function main() {
  if (!fs.existsSync(outputsDir)) {
    throw new Error(`Outputs folder missing: ${outputsDir}. Run python demo first.`);
  }
  if (!fs.existsSync(docsFigDir)) {
    throw new Error(`Docs figure folder missing: ${docsFigDir}`);
  }
  if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
  }

  const stats = loadStats();

  const ppt = new PptxGenJS();
  ppt.layout = 'LAYOUT_WIDE';
  ppt.author = NAME;
  ppt.company = 'FAST NUCES';
  ppt.subject = 'Attention-Guided Evolutionary Art';
  ppt.title = 'Attention-Guided Evolutionary Art';
  ppt.theme = {
    headFontFace: FONT.heading,
    bodyFontFace: FONT.body,
    lang: 'en-US',
  };

  addDeck(ppt, stats);

  await ppt.writeFile({ fileName: pptDistPath });
  fs.copyFileSync(pptDistPath, pptRootPath);

  console.log(`[pptx] generated: ${pptDistPath}`);
  console.log(`[pptx] copied to root: ${pptRootPath}`);
}

main().catch((err) => {
  console.error('[pptx] generation failed:', err);
  process.exit(1);
});
