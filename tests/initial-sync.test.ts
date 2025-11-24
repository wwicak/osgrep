import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { FileSystem } from "../src/lib/file";
import type { IndexFileOptions, Store } from "../src/lib/store";
import { initialSync } from "../src/utils";

class FakeStore implements Store {
  indexed: Array<{ storeId: string; externalId?: string }> = [];
  deleted: string[] = [];
  ftsCount = 0;
  vecCount = 0;

  async *listFiles(_storeId: string) {
    // empty store
    yield* [];
  }

  async indexFile(
    storeId: string,
    _content: string | File,
    options: IndexFileOptions,
  ) {
    this.indexed.push({ storeId, externalId: options.external_id });
  }

  async deleteFile(_storeId: string, externalId: string) {
    this.deleted.push(externalId);
  }

  async search(
    _storeId: string,
    _query: string,
    _top_k?: number,
    _search_options?: { rerank?: boolean },
  ) {
    throw new Error("not implemented");
  }

  async retrieve(_storeId: string) {
    throw new Error("not implemented");
  }

  async create(_options) {
    throw new Error("not implemented");
  }

  async ask(
    _storeId: string,
    _question: string,
    _top_k?: number,
    _search_options?: { rerank?: boolean },
  ) {
    throw new Error("not implemented");
  }

  async getInfo(_storeId: string) {
    throw new Error("not implemented");
  }

  async createFTSIndex(_storeId: string) {
    this.ftsCount += 1;
  }

  async createVectorIndex(_storeId: string) {
    this.vecCount += 1;
  }

  // Unused by these tests
  async *listStoreIds() {
    yield* [];
  }

  async close() {}
}

class StubFileSystem implements FileSystem {
  constructor(
    private files: string[],
    private ignored: Set<string> = new Set(),
  ) {}

  *getFiles(_dirRoot: string): Generator<string> {
    yield* this.files;
  }

  isIgnored(filePath: string): boolean {
    return this.ignored.has(filePath);
  }

  loadOsgrepignore(): void {}
}

describe("initialSync edge cases", () => {
  let tempRoot: string;

  beforeEach(async () => {
    tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "osgrep-sync-"));
  });

  afterEach(async () => {
    await fs.rm(tempRoot, { recursive: true, force: true });
  });

  it("handles an empty repository and store without indexing", async () => {
    const store = new FakeStore();
    const fsStub = new StubFileSystem([]);

    const result = await initialSync(store, fsStub, "store", tempRoot, false);

    expect(result.total).toBe(0);
    expect(result.indexed).toBe(0);
    expect(store.indexed.length).toBe(0);
    expect(store.ftsCount).toBe(0);
    expect(store.vecCount).toBe(0);
  });

  it("skips indexing when every file is ignored", async () => {
    const store = new FakeStore();
    const ignoredFile = path.join(tempRoot, "ignored.ts");
    await fs.writeFile(ignoredFile, "content");

    const fsStub = new StubFileSystem([ignoredFile], new Set([ignoredFile]));

    const result = await initialSync(store, fsStub, "store", tempRoot, false);

    expect(result.total).toBe(0);
    expect(result.indexed).toBe(0);
    expect(store.indexed.length).toBe(0);
    expect(store.ftsCount).toBe(0);
    expect(store.vecCount).toBe(0);
  });
});
