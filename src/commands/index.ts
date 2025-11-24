import { Command } from "commander";
import { createFileSystem, createStore } from "../lib/context";
import { ensureSetup } from "../lib/setup-helpers";
import type { Store } from "../lib/store";
import { ensureStoreExists } from "../lib/store-helpers";
import { getAutoStoreId } from "../lib/store-resolver";
import {
  createIndexingSpinner,
  formatDryRunSummary,
} from "../lib/sync-helpers";
import { initialSync, MetaStore } from "../utils";

const PROFILE_ENABLED =
  process.env.OSGREP_PROFILE === "1" || process.env.OSGREP_PROFILE === "true";

export const index = new Command("index")
  .description("Index the current directory and create searchable store")
  .option(
    "-d, --dry-run",
    "Dry run the indexing process (no actual file syncing)",
    false,
  )
  .option(
    "-p, --path <dir>",
    "Path to index (defaults to current directory)",
    "",
  )
  .action(async (_args, cmd) => {
    const options: { store?: string; dryRun: boolean; path: string } =
      cmd.optsWithGlobals();

    let store: Store | null = null;
    try {
      await ensureSetup();
      store = await createStore();

      // Auto-detect store ID if not explicitly provided
      const indexRoot = options.path || process.cwd();
      const storeId = options.store || getAutoStoreId(indexRoot);

      await ensureStoreExists(store, storeId);
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
        ],
      });
      const metaStore = new MetaStore();

      const { spinner, onProgress } = createIndexingSpinner(
        indexRoot,
        "Indexing...",
      );
      try {
        try {
          await store.retrieve(storeId);
        } catch {
          await store.create({
            name: storeId,
            description: "osgrep local index",
          });
        }
        const result = await initialSync(
          store,
          fileSystem,
          storeId,
          indexRoot,
          options.dryRun,
          onProgress,
          metaStore,
        );

        if (options.dryRun) {
          spinner.succeed(
            `Dry run complete (${result.processed}/${result.total}) • would have indexed ${result.indexed}`,
          );
          if (PROFILE_ENABLED && typeof store.getProfile === "function") {
            console.log("[profile] local store:", store.getProfile());
          }
          console.log(
            formatDryRunSummary(result, {
              actionDescription: "would have indexed",
              includeTotal: true,
            }),
          );
          return; // Let Node exit naturally
        }

        // Wait for all indexing to complete
        while (true) {
          const info = await store.getInfo(storeId);
          spinner.text = `Indexing ${info.counts.pending + info.counts.in_progress} file(s)`;
          if (info.counts.pending === 0 && info.counts.in_progress === 0) {
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }

        spinner.succeed(
          `Indexing complete (${result.processed}/${result.total}) • indexed ${result.indexed}`,
        );
        if (PROFILE_ENABLED && typeof store.getProfile === "function") {
          console.log("[profile] local store:", store.getProfile());
        }
      } catch (e) {
        spinner.fail("Indexing failed");
        throw e;
      } finally {
        // Always clean up the store
        if (store && typeof store.close === "function") {
          try {
            await store.close();
          } catch (err) {
            console.error("Failed to close store:", err);
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      console.error("Failed to index:", message);
      process.exitCode = 1;
    }
  });
