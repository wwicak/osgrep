import { createReadStream } from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

describe.sequential("LocalStore search integration", () => {
  const storeId = "integration-store";
  const originalHome = process.env.HOME;
  const originalUserProfile = process.env.USERPROFILE;
  let tempHome: string;
  let repoDir: string;
  let dataDir: string;
  let LocalStore: typeof import("../../src/lib/local-store").LocalStore;
  let store: InstanceType<typeof LocalStore>;

  // Helper to fix schema mismatch (undefined vs "") for LanceDB
  const sanitizeRecords = (records: any[]) => {
    return records.map((r) => ({
      ...r,
      context_prev: r.context_prev ?? "",
      context_next: r.context_next ?? "",
    }));
  };

  beforeAll(async () => {
    tempHome = await fs.mkdtemp(path.join(os.tmpdir(), "osgrep-int-"));
    process.env.HOME = tempHome;
    process.env.USERPROFILE = tempHome;
    dataDir = path.join(os.homedir(), ".osgrep", "data");

    ({ LocalStore } = await import("../../src/lib/local-store"));
    store = new LocalStore();

    // Stub embedding calls to avoid pulling real models while keeping LanceDB queries real
    const workerManager = (store as any).workerManager;
    vi.spyOn(workerManager, "getEmbeddings").mockImplementation(
      async (texts: string[]) =>
        texts.map((_text, idx) => Array(384).fill(idx + 1)),
    );
    vi.spyOn(workerManager, "getEmbedding").mockResolvedValue(
      Array(384).fill(0.5),
    );
    vi.spyOn(workerManager, "rerank").mockImplementation(
      async (_query, docs) => docs.map((_doc, idx) => 1 - idx * 0.05),
    );

    // Force initialization of the underlying LanceDB connection using the temp HOME
    await (store as any).getDb();

    repoDir = path.join(tempHome, "repo");
    await fs.mkdir(repoDir, { recursive: true });

    const mainPath = path.join(repoDir, "main.ts");
    const utilsPath = path.join(repoDir, "utils.ts");

    await fs.writeFile(
      mainPath,
      `function login() {
  return "ok";
}
`,
      "utf-8",
    );

    await fs.writeFile(
      utilsPath,
      `export function helper() {
  return "utils";
}
`,
      "utf-8",
    );

    await store.create({ name: storeId });

    // Update: Capture returned records, sanitize them, and manually insert
    const mainRecords = await store.indexFile(
      storeId,
      createReadStream(mainPath),
      { metadata: { path: "main.ts", hash: "hash-main" } },
    );
    await store.insertBatch(storeId, sanitizeRecords(mainRecords));

    // Update: Capture returned records, sanitize them, and manually insert
    const utilsRecords = await store.indexFile(
      storeId,
      createReadStream(utilsPath),
      { metadata: { path: "utils.ts", hash: "hash-utils" } },
    );
    await store.insertBatch(storeId, sanitizeRecords(utilsRecords));
  }, 30_000);

  afterAll(async () => {
    await store?.close();
    if (tempHome) {
      await fs.rm(tempHome, { recursive: true, force: true });
    }
    vi.restoreAllMocks();
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;

    if (originalUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = originalUserProfile;
  });

  it("writes LanceDB data under the temporary home directory", async () => {
    const stat = await fs.stat(dataDir);
    expect(stat.isDirectory()).toBe(true);
  });

  it.skip("returns indexed chunks when searching for login", async () => {
    const res = await store.search(storeId, "login", 5);
    const loginChunk = res.data.find((c) => c.metadata?.path === "main.ts");

    expect(loginChunk).toBeDefined();
    expect(loginChunk?.text.toLowerCase()).toContain("login");
    expect((loginChunk?.score ?? 0)).toBeGreaterThan(0);
  });

  it.skip("applies path filters to exclude other files", async () => {
    const res = await store.search(
      storeId,
      "function",
      10,
      undefined,
      {
        all: [
          { key: "path", operator: "starts_with", value: "main.ts" },
        ],
      } as any,
    );

    expect(res.data.length).toBeGreaterThan(0);
    expect(res.data.some((c) => c.metadata?.path === "utils.ts")).toBe(false);
    expect(
      res.data.every((c) => (c.metadata?.path || "").startsWith("main.ts")),
    ).toBe(true);
  });
});
