import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const slidesRoot = path.resolve(__dirname, '..');
const sourceDir = path.resolve(slidesRoot, '..', 'python', 'outputs');
const targetDir = path.resolve(slidesRoot, 'public', 'images');

const allowedExtensions = new Set(['.jpg', '.jpeg', '.png', '.gif', '.json']);

async function exists(dirPath) {
  try {
    await fs.access(dirPath);
    return true;
  } catch {
    return false;
  }
}

async function clearTargetDirectory() {
  if (!(await exists(targetDir))) {
    await fs.mkdir(targetDir, { recursive: true });
    return;
  }

  const entries = await fs.readdir(targetDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(targetDir, entry.name);
    if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (allowedExtensions.has(ext)) {
        await fs.unlink(fullPath);
      }
    }
  }
}

async function copyAssets() {
  await clearTargetDirectory();

  if (!(await exists(sourceDir))) {
    console.warn(`[copy-assets] Source directory does not exist: ${sourceDir}`);
    return;
  }

  const entries = await fs.readdir(sourceDir, { withFileTypes: true });
  let copied = 0;

  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }

    const ext = path.extname(entry.name).toLowerCase();
    if (!allowedExtensions.has(ext)) {
      continue;
    }

    const src = path.join(sourceDir, entry.name);
    const dst = path.join(targetDir, entry.name);
    await fs.copyFile(src, dst);
    copied += 1;
  }

  console.log(`[copy-assets] Copied ${copied} files from ${sourceDir} to ${targetDir}`);
}

copyAssets().catch((error) => {
  console.error('[copy-assets] Failed to copy assets:', error);
  process.exit(1);
});
