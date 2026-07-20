import Foundation
import XCTest
@testable import Gist

/// The Swift Unigram tokenizer must reproduce the training (model2vec) tokenizer's
/// ids exactly — the whole semantic stream depends on identical token ids.
final class TokenizerTests: XCTestCase {
    struct Case: Decodable { let text: String; let ids: [Int] }

    func testMatchesPythonOracle() throws {
        let binURL = try XCTUnwrap(Bundle.module.url(forResource: "gist_tokenizer", withExtension: "bin"))
        let tok = try XCTUnwrap(Tokenizer(bytes: [UInt8](try Data(contentsOf: binURL))))

        let oracleURL = try XCTUnwrap(Bundle.module.url(forResource: "gist-sdk-oracle", withExtension: "json"))
        let cases = try JSONDecoder().decode([Case].self, from: try Data(contentsOf: oracleURL))

        for c in cases {
            XCTAssertEqual(tok.encode(c.text), c.ids, "tokenizer mismatch for \"\(c.text)\"")
        }
    }
}
