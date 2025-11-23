import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import chokidar, { type FSWatcher } from "chokidar";
import { Command } from "commander";
import { createFileSystem, createStore } from "../lib/context";
import { ensureSetup } from "../lib/setup-helpers";
import type { Store } from "../lib/store";
import { ensureStoreExists, isStoreEmpty } from "../lib/store-helpers";
import { getAutoStoreId } from "../lib/store-resolver";
import {
  clearServerLock,
  computeBufferHash,
  debounce,
  formatDenseSnippet,
  indexFile,
  initialSync,
  isIndexablePath,
  MetaStore,
  readServerLock,
  writeServerLock,
} from "../utils";

type PendingAction = "upsert" | "delete";

function toDenseResults(
  storeRoot: string,
  data: Array<{
    score: number;
    text?: string | null;
    metadata?: Record<string, unknown>;
    generated_metadata?: { start_line?: number | null };
  }>,
) {
  const root = path.resolve(storeRoot);
  return data.map((item) => {
    const rawPath =
      typeof item.metadata?.path === "string"
        ? (item.metadata.path as string)
        : "";
    const relPath = rawPath
      ? path.relative(root, rawPath) || rawPath
      : "unknown";
    const snippet = formatDenseSnippet(item.text ?? "");
    return {
      path: relPath,
      score: Number(item.score.toFixed(3)),
      content: snippet,
    };
  });
}

async function createWatcher(
  store: Store,
  storeId: string,
  root: string,
  metaStore: MetaStore,
): Promise<FSWatcher> {
  const fileSystem = createFileSystem({
    ignorePatterns: [
      "*.lock",
      "*.bin",
      "*.ipynb",
      "*.pyc",
      "pnpm-lock.yaml",
      "package-lock.json",
      "yarn.lock",
      "bun.lockb",
      ".osgrep/**",
    ],
  });

  const pending = new Map<string, PendingAction>();

  const processPending = debounce(async () => {
    const actions = Array.from(pending.entries());
    pending.clear();
    for (const [filePath, action] of actions) {
      if (action === "delete") {
        try {
          await store.deleteFile(storeId, filePath);
          metaStore.delete(filePath);
          await metaStore.save();
        } catch (err) {
          console.error("Failed to delete file from store:", err);
        }
        continue;
      }

      if (fileSystem.isIgnored(filePath, root) || !isIndexablePath(filePath)) {
        continue;
      }

      try {
        const buffer = await fs.promises.readFile(filePath);
        if (buffer.length === 0) continue;
        const hash = computeBufferHash(buffer);
        const didIndex = await indexFile(
          store,
          storeId,
          filePath,
          path.basename(filePath),
          metaStore,
          undefined,
          buffer,
          hash,
        );
        if (didIndex) {
          metaStore.set(filePath, hash);
          await metaStore.save();
        }
      } catch (err) {
        console.error("Failed to index changed file:", err);
      }
    }
  }, 300);

  const watcher = chokidar.watch(root, {
    ignoreInitial: true,
    persistent: true,
    ignored: (watchedPath) =>
      fileSystem.isIgnored(watchedPath.toString(), root) ||
      watchedPath.toString().includes(`${path.sep}.git${path.sep}`) ||
      watchedPath.toString().includes(`${path.sep}.osgrep${path.sep}`),
  });

  watcher
    .on("add", (filePath) => {
      pending.set(path.resolve(filePath), "upsert");
      processPending();
    })
    .on("change", (filePath) => {
      pending.set(path.resolve(filePath), "upsert");
      processPending();
    })
    .on("unlink", (filePath) => {
      pending.set(path.resolve(filePath), "delete");
      processPending();
    });

  return watcher;
}

async function respondJson(
  res: http.ServerResponse,
  status: number,
  payload: object,
) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

export const serve = new Command("serve")
  .description("Run osgrep as a background server with live indexing")
  .option(
    "-p, --port <port>",
    "Port to listen on",
    process.env.OSGREP_PORT || "4444",
  )
  .action(async (_args, cmd) => {
    const options: { port: string; store?: string } = cmd.optsWithGlobals();
    const port = parseInt(options.port, 10);
    const root = process.cwd();

    let store: Store | null = null;
    let watcher: FSWatcher | null = null;
    const metaStore = new MetaStore();

    const shutdown = async () => {
      try {
        await clearServerLock(root);
      } catch (err) {
        console.error("Failed to clear server lock:", err);
      }
      try {
        await watcher?.close();
      } catch (err) {
        console.error("Failed to close watcher:", err);
      }
      if (store && typeof store.close === "function") {
        try {
          await store.close();
        } catch (err) {
          console.error("Failed to close store:", err);
        }
      }
      process.exit(0);
    };

    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
    process.on("exit", async () => {
      await clearServerLock(root);
    });

    try {
      await ensureSetup({ silent: true });
      await metaStore.load();
      store = await createStore();
      const storeId = options.store || getAutoStoreId(root);
      await ensureStoreExists(store, storeId);

      const empty = await isStoreEmpty(store, storeId);
      if (empty) {
        const fileSystem = createFileSystem({
          ignorePatterns: [
            "*.lock",
            "*.bin",
            "*.ipynb",
            "*.pyc",
            "pnpm-lock.yaml",
            "package-lock.json",
            "yarn.lock",
            "bun.lockb",
            ".osgrep/**",
          ],
        });
        console.log("Store empty, performing initial index...");
        await initialSync(
          store,
          fileSystem,
          storeId,
          root,
          false,
          undefined,
          metaStore,
        );
        console.log("Initial index complete.");
      }

      watcher = await createWatcher(store, storeId, root, metaStore);

      const server = http.createServer(async (req, res) => {
        if (!req.url) {
          return respondJson(res, 400, { error: "Invalid request" });
        }

        const url = new URL(req.url, `http://localhost:${port}`);
        if (req.method === "GET" && url.pathname === "/health") {
          return respondJson(res, 200, { status: "ready" });
        }

        if (req.method === "POST" && url.pathname === "/search") {
          const chunks: Buffer[] = [];
          req.on("data", (c) => chunks.push(c));
          req.on("end", async () => {
            try {
              const bodyRaw = Buffer.concat(chunks).toString("utf-8");
              const body = bodyRaw ? JSON.parse(bodyRaw) : {};
              const query = typeof body.query === "string" ? body.query : "";
              if (!query) {
                return respondJson(res, 400, { error: "query is required" });
              }
              const limit =
                typeof body.limit === "number" && !Number.isNaN(body.limit)
                  ? body.limit
                  : 25;
              const rerank = body.rerank === false ? false : true;

              const searchPath =
                typeof body.path === "string" && body.path.length > 0
                  ? path.normalize(
                      path.isAbsolute(body.path)
                        ? body.path
                        : path.join(root, body.path),
                    )
                  : root;

              const filters =
                body.filters && typeof body.filters === "object"
                  ? body.filters
                  : {
                      all: [
                        {
                          key: "path",
                          operator: "starts_with",
                          value: searchPath,
                        },
                      ],
                    };

              const results = await store!.search(
                storeId,
                query,
                limit,
                { rerank },
                filters,
              );
              const dense = toDenseResults(root, results.data);
              return respondJson(res, 200, { results: dense });
            } catch (err) {
              console.error("Search handler failed:", err);
              return respondJson(res, 500, { error: "search_failed" });
            }
          });
          return;
        }

        return respondJson(res, 404, { error: "not_found" });
      });

      server.listen(port, async () => {
        await writeServerLock(port, process.pid, root);
        const lock = await readServerLock(root);
        console.log(
          `osgrep serve listening on port ${port} (lock: ${lock?.pid ?? "n/a"})`,
        );
      });
    } catch (error) {
      console.error(
        "Failed to start osgrep server:",
        error instanceof Error ? error.message : "Unknown error",
      );
      await shutdown();
    }
  });
