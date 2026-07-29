package ai.desertant.gist

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests for the Android binding, exercising the real on-device path
 * via JNI: platform JSON via CHostBridge, LiteRT inference, and the
 * static-stdlib runtime. The bundled model comes from the
 * `gist-tflite-resources` androidTest dependency.
 */
@RunWith(AndroidJUnit4::class)
class GistTest {
    private lateinit var gist: Gist

    @Before fun setUp() { gist = Gist.bundled() }
    @After fun tearDown() { gist.close() }

    @Test fun classifiesEnglishPhrase() = runTest {
        val topics = gist.classify("How to start a podcast with just your iPhone", topK = 5)
        assertTrue("expected topics", topics.isNotEmpty())
        assertTrue("got ${topics.map { it.slug }}",
            topics.any { it.slug in listOf("technology", "creator-economy") })
    }

    @Test fun ranksByConfidence() = runTest {
        val topics = gist.classify("Cómo invertir en fondos indexados", topK = 3)
        assertTrue("expected topics", topics.isNotEmpty())
        assertTrue(topics.zipWithNext().all { (a, b) -> a.score >= b.score })
        assertTrue(topics.all { it.score in 0.0..1.0 })
    }

    @Test fun scoresCoverFullTaxonomy() = runTest {
        val scores = gist.scores("The latest election results and what they mean")
        assertEquals("36-topic distribution", 36, scores.size)
        assertTrue(scores.values.all { it in 0.0..1.0 })
    }
}
