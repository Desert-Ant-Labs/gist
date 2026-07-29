// Browser half of the platform seam for the universal WebAssembly entry
// (browser.js). Bundlers resolve this file through the "browser" import
// condition of `#platform` (see package.json "imports"), so none of the
// node-only code in platform-node.js ever enters the browser module graph.
// The shared instantiation logic lives in @desert-ant-labs/core.
import { browserSetup, browserWasmDir, browserReadModelSource, browserCacheRoot } from "@desert-ant-labs/core";

export function setupCore() {
  return browserSetup({
    hostGlobal: "__GistHost",
    exportsGlobal: "__GistExports",
    init: () => import("./dist/index.js"),
  });
}

export const defaultWasmDir = browserWasmDir;
export const readModelSource = browserReadModelSource;
export const defaultCacheRoot = browserCacheRoot;
