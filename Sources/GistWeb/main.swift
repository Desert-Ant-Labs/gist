// WebAssembly entry point for the gist Node/web package. Mirrors the Swift SDK.
// The JS host must set `globalThis.__GistHost` (an async LiteRT.js session, see
// ModelLoading.swift) before the first call. After start, the module exposes:
//
//     globalThis.__GistExports = {
//       load(cacheRoot, directory?, onProgress?)  -> Promise<boolean>,
//       scores(text)                              -> Promise<{slug: prob, ...}>,
//       classify(text, topK?, threshold?)         -> Promise<[{slug, name, score}]>,
//     }
//
// packages/gist-node wraps this in the public typed API; nothing else should
// touch these globals.
#if os(WASI)
import JavaScriptEventLoop
import JavaScriptKit
@_spi(GistBindings) import Gist

JavaScriptEventLoop.installGlobalExecutor()

private nonisolated(unsafe) var tagger: Gist?
private func instance() throws -> Gist {
    guard let tagger else { throw GistError.resourceMissing }
    return tagger
}

let loadFn = JSClosure { args in
    let cacheRoot = args.first?.string.flatMap { $0.isEmpty ? nil : $0 }
    let directory = (args.count > 1 ? args[1].string : nil).flatMap { $0.isEmpty ? nil : $0 }
    let onProgress: JSFunction? = args.count > 2 ? args[2].function : nil
    let gist = Gist(directory: directory, cacheRoot: cacheRoot)
    return JSPromise { resolve in
        Task {
            do {
                try await gist.download { fraction in if let onProgress { _ = onProgress(fraction) } }
                tagger = gist
                resolve(.success(.boolean(true)))
            } catch {
                resolve(.failure(.string(String(describing: error))))
            }
        }
    }.jsValue
}

let scoresFn = JSClosure { args in
    let text = args.first?.string ?? ""
    return JSPromise { resolve in
        Task {
            do {
                let scores = try await instance().scores(of: text)
                let out = JSObject.global.Object.function!.new()
                for (slug, prob) in scores { out[slug] = .number(prob) }
                resolve(.success(.object(out)))
            } catch {
                resolve(.failure(.string(String(describing: error))))
            }
        }
    }.jsValue
}

let classifyFn = JSClosure { args in
    let text = args.first?.string ?? ""
    let topK = args.count > 1 ? Int(args[1].number ?? 3) : 3
    let threshold = args.count > 2 ? args[2].number : nil
    return JSPromise { resolve in
        Task {
            do {
                let topics = try await instance().classify(text, topK: topK, threshold: threshold)
                let arr = JSObject.global.Array.function!.new()
                for (i, t) in topics.enumerated() {
                    let o = JSObject.global.Object.function!.new()
                    o.slug = .string(t.slug)
                    o.name = .string(t.name)
                    o.score = .number(t.score)
                    arr[i] = .object(o)
                }
                resolve(.success(.object(arr)))
            } catch {
                resolve(.failure(.string(String(describing: error))))
            }
        }
    }.jsValue
}

let exports = JSObject.global.Object.function!.new()
exports.load = .object(loadFn)
exports.scores = .object(scoresFn)
exports.classify = .object(classifyFn)
JSObject.global.__GistExports = .object(exports)
#endif
