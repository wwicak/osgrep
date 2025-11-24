import * as fs from "node:fs";
import * as path from "node:path";
import { Worker } from "node:worker_threads";
import { v4 as uuidv4 } from "uuid";
import { VECTOR_CACHE_MAX, WORKER_TIMEOUT_MS } from "../config";
import { LRUCache } from "./lru";

type WorkerRequest =
  | { id: string; text: string }
  | { id: string; texts: string[] }
  | { id: string; rerank: { query: string; documents: string[] } };

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  payload: WorkerRequest;
  timeoutId?: NodeJS.Timeout;
  workerIndex: number;
};

export class WorkerManager {
  // Initialize with nulls to preserve index positions during staggering
  private workers: Array<Worker | null> = [];
  private vectorCache = new LRUCache<string, number[]>(VECTOR_CACHE_MAX);
  private pendingRequests = new Map<string, PendingRequest>();
  private restartInFlight = new Map<number, Promise<void>>();
  private isClosing = false;
  private readonly MAX_WORKER_RSS = 6 * 1024 * 1024 * 1024;
  private nextWorkerIndex = 0;

  constructor() {
    // Fire and forget the async initialization
    this.initializeWorkers().catch((err) =>
      console.error("Failed to initialize worker pool:", err),
    );
  }

  private getWorkerConfig(): { workerPath: string; execArgv: string[] } {
    const tsWorkerPath = path.join(__dirname, "worker.ts");
    const jsWorkerPath = path.join(__dirname, "worker.js");
    const hasTsWorker = fs.existsSync(tsWorkerPath);
    const hasJsWorker = fs.existsSync(jsWorkerPath);
    const runningTs = path.extname(__filename) === ".ts";
    const isDev = (runningTs && hasTsWorker) || (hasTsWorker && !hasJsWorker);

    if (isDev) {
      return { workerPath: tsWorkerPath, execArgv: ["-r", "ts-node/register"] };
    }
    return { workerPath: jsWorkerPath, execArgv: [] };
  }

  private getWorkerCount(): number {
    // Force single worker in test or when explicitly requested
    if (
      process.env.OSGREP_SINGLE_WORKER === "1" ||
      process.env.NODE_ENV === "test"
    ) {
      return 1;
    }

    // 1. Check environment variable override first
    if (process.env.OSGREP_WORKER_COUNT) {
      const count = parseInt(process.env.OSGREP_WORKER_COUNT, 10);
      if (!Number.isNaN(count) && count > 0) {
        return Math.min(4, count);
      }
    }
    
    // 2. PRODUCTION SAFETY: Default to 1.
    // onnxruntime-node has thread-safety issues in Node.js worker_threads on macOS/Linux.
    // Defaulting to 1 ensures stability for general users.
    return 1; 
  }

  private async initializeWorkers() {
    const workerCount = this.getWorkerCount();
    // Pre-fill array so indices exist
    this.workers = new Array(workerCount).fill(null);

    if (workerCount > 1) {
        console.log(`[WorkerManager] Initializing pool with ${workerCount} workers...`);
    }

    for (let i = 0; i < workerCount; i++) {
      if (this.isClosing) break;
      
      this.createWorker(i);

      // Stagger creation to try and reduce V8 race conditions, though 
      // single-threaded default is the real fix.
      if (i < workerCount - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
  }

  private createWorker(index: number) {
    const { workerPath, execArgv } = this.getWorkerConfig();
    const worker = new Worker(workerPath, { execArgv });
    this.workers[index] = worker;

    worker.on("message", (message) => {
      const { id, vector, vectors, scores, error, memory } = message;
      const pending = this.pendingRequests.get(id);

      // Only resolve if this worker actually handled it
      if (pending && pending.workerIndex === index) {
        if (pending.timeoutId) clearTimeout(pending.timeoutId);

        if (error) {
          pending.reject(new Error(error));
        } else if (vectors !== undefined) {
          pending.resolve(vectors);
        } else if (scores !== undefined) {
          pending.resolve(scores);
        } else {
          pending.resolve(vector);
        }
        this.pendingRequests.delete(id);
      }

      if (memory && memory.rss > this.MAX_WORKER_RSS) {
        console.warn(
          `Worker ${index} memory usage high (${Math.round(memory.rss / 1024 / 1024)}MB). Restarting...`,
        );
        void this.restartWorker(index, "memory limit exceeded");
      }
    });

    worker.on("error", (err) => {
      console.error(`Worker ${index} error:`, err);
      void this.restartWorker(index, `worker error: ${err.message}`);
    });

    worker.on("exit", (code) => {
      if (code !== 0 && !this.isClosing) {
        console.error(`Worker ${index} exited with code ${code}`);
        void this.restartWorker(index, `worker exit: code ${code}`);
      }
    });
  }

  private rejectPendingForWorker(workerIndex: number, error: Error) {
    for (const [id, pending] of this.pendingRequests.entries()) {
      if (pending.workerIndex !== workerIndex) continue;
      if (pending.timeoutId) clearTimeout(pending.timeoutId);
      pending.reject(error);
      this.pendingRequests.delete(id);
    }
  }

  private async restartWorker(workerIndex: number, reason?: string) {
    const inFlight = this.restartInFlight.get(workerIndex);
    if (inFlight) {
      return inFlight;
    }

    this.rejectPendingForWorker(
      workerIndex,
      new Error(`Worker restarting: ${reason || "unknown reason"}`),
    );

    const restartPromise = (async () => {
      try {
        const worker = this.workers[workerIndex];
        if (worker) {
          await worker.terminate();
        }
      } catch (err) {
        console.error("Failed to terminate worker cleanly:", err);
      }
      this.workers[workerIndex] = null;
      // Small cool-down before restarting a crashed worker
      await new Promise(resolve => setTimeout(resolve, 500));
      this.createWorker(workerIndex);
      if (reason) {
        console.warn(`Worker ${workerIndex} restarted due to ${reason}`);
      }
    })();

    this.restartInFlight.set(workerIndex, restartPromise);
    await restartPromise;
    this.restartInFlight.delete(workerIndex);
  }

  private async getNextWorker(): Promise<{ worker: Worker; index: number }> {
    // Wait loop: if we are just starting up, we might need to wait for Worker 0
    const maxRetries = 50; // 5 seconds max wait for first boot
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      // Filter for actual active workers (not null)
      const activeIndices = this.workers
        .map((w, i) => (w !== null ? i : -1))
        .filter((i) => i !== -1);

      if (activeIndices.length > 0) {
        // Round robin among *active* workers
        const selectedIndex = activeIndices[this.nextWorkerIndex % activeIndices.length];
        this.nextWorkerIndex = (this.nextWorkerIndex + 1) % activeIndices.length; // Advance logic
        
        const worker = this.workers[selectedIndex];
        if (worker) {
            return { worker, index: selectedIndex };
        }
      }
      
      // Wait 100ms before retrying
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    throw new Error("No workers available (timeout waiting for pool)");
  }

  private async sendToWorker<T>(
    buildPayload: (id: string) => WorkerRequest,
  ): Promise<T> {
    const { worker, index } = await this.getNextWorker();

    return new Promise((resolve, reject) => {
      const id = uuidv4();
      const payload = buildPayload(id);

      const timeoutId = setTimeout(() => {
        const pending = this.pendingRequests.get(id);
        if (pending) {
          this.pendingRequests.delete(id);
          reject(
            new Error(`Worker request timed out after ${WORKER_TIMEOUT_MS}ms`),
          );
        }
      }, WORKER_TIMEOUT_MS);

      this.pendingRequests.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject: reject as (reason: unknown) => void,
        payload,
        timeoutId,
        workerIndex: index,
      });
      worker.postMessage(payload);
    });
  }

  async getEmbeddings(texts: string[]): Promise<number[][]> {
    const neededIndices: number[] = [];
    const neededTexts: string[] = [];
    const results: number[][] = new Array(texts.length);

    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const cached = this.vectorCache.get(text);
      if (cached !== undefined) {
        results[i] = cached;
      } else {
        neededIndices.push(i);
        neededTexts.push(text);
      }
    }

    if (neededTexts.length === 0) return results;

    const computedVectors = await this.sendToWorker<number[][]>((id) => ({
      id,
      texts: neededTexts,
    }));

    for (let i = 0; i < computedVectors.length; i++) {
      const originalIndex = neededIndices[i];
      const vector = computedVectors[i];
      const text = neededTexts[i];

      this.vectorCache.set(text, vector);
      results[originalIndex] = vector;
    }

    return results;
  }

  async getEmbedding(text: string): Promise<number[]> {
    const results = await this.getEmbeddings([text]);
    return results[0];
  }

  async rerank(query: string, documents: string[]): Promise<number[]> {
    return this.sendToWorker<number[]>((id) => ({
      id,
      rerank: { query, documents },
    }));
  }

  async close(): Promise<void> {
    this.isClosing = true;
    try {
      for (const worker of this.workers) {
        worker?.postMessage({ type: "shutdown" });
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
      await Promise.all(
        this.workers.map(async (worker) => {
          if (!worker) return;
          try {
            await worker.terminate();
          } catch {
            // Silent cleanup
          }
        }),
      );
    } catch (err) {
      // Silent cleanup
    }
  }
}
