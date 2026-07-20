import Foundation
import XCTest
@testable import Gist

/// The semantic (embedding pool) and lexical (hashed n-gram) streams must match
/// the Python/gist-js reference exactly — they are the head's input. (The head
/// itself is validated separately: the LiteRT .tflite is bit-identical to ONNX.)
final class PipelineTests: XCTestCase {
    struct Case: Decodable { let text: String; let emb: [Float]; let ngram_nz: [String: Float] }

    func testSemanticAndLexicalStreams() throws {
        let tokURL = try XCTUnwrap(Bundle.module.url(forResource: "gist_tokenizer", withExtension: "bin"))
        let tok = try XCTUnwrap(Tokenizer(bytes: [UInt8](try Data(contentsOf: tokURL))))

        let embURL = try XCTUnwrap(Bundle.module.url(forResource: "gist_embedding", withExtension: "i8"))
        let embMetaURL = try XCTUnwrap(Bundle.module.url(forResource: "gist_embedding", withExtension: "json"))
        let embedding = try Embedding(
            rows: [UInt8](try Data(contentsOf: embURL)),
            metaJSON: String(decoding: try Data(contentsOf: embMetaURL), as: UTF8.self))

        let oracleURL = try XCTUnwrap(Bundle.module.url(forResource: "gist-feature-oracle", withExtension: "json"))
        let cases = try JSONDecoder().decode([Case].self, from: try Data(contentsOf: oracleURL))

        for c in cases {
            // semantic stream
            let sem = embedding.pool(ids: tok.encode(c.text))
            XCTAssertEqual(sem.count, c.emb.count, "emb dim for \"\(c.text)\"")
            var maxDiff: Float = 0
            for (a, b) in zip(sem, c.emb) { maxDiff = max(maxDiff, abs(a - b)) }
            XCTAssertLessThan(maxDiff, 1e-3, "semantic embedding drift for \"\(c.text)\"")

            // lexical stream
            let ng = NGrams.features(c.text, dim: 8192)
            for (idxStr, expected) in c.ngram_nz {
                let i = Int(idxStr)!
                XCTAssertEqual(ng[i], expected, accuracy: 1e-4, "n-gram[\(i)] for \"\(c.text)\"")
            }
            let nz = ng.enumerated().filter { $0.element != 0 }.count
            XCTAssertEqual(nz, c.ngram_nz.count, "n-gram nonzero count for \"\(c.text)\"")
        }
    }
}
