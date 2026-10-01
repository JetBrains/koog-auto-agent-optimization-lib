package ai.koog.agents.optimization.utils.messages

import ai.koog.prompt.message.AttachmentContent
import ai.koog.prompt.message.AttachmentSource
import ai.koog.prompt.message.MessagePart
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Pins what [render] keeps and what it drops, so the two lossy branches (attachment payloads and
 * non-text tool-result parts) stay visible rather than turning into a silently truncated trace.
 */
internal class MessagePartsRenderTest {

    @Test
    fun `tool result renders its text parts and drops the rest`() {
        val result = MessagePart.Tool.Result(
            tool = "read_file",
            parts = listOf(
                MessagePart.Text("before"),
                MessagePart.Attachment(imageSource()),
                MessagePart.Text("after"),
            ),
        )

        // `Result.output` concatenates text parts only -- the attachment is what the warn-log reports.
        assertEquals("beforeafter", result.render())
    }

    @Test
    fun `attachment renders as a placeholder`() {
        assertEquals("<attachment>", MessagePart.Attachment(imageSource()).render())
    }

    @Test
    fun `text tool call and reasoning render in full`() {
        assertEquals("hi", MessagePart.Text("hi").render())
        assertEquals("""t({"a":1})""", MessagePart.Tool.Call(tool = "t", args = """{"a":1}""").render())
        assertEquals("one\ntwo", MessagePart.Reasoning(listOf("one", "two")).render())
    }

    private fun imageSource() = AttachmentSource.Image(
        content = AttachmentContent.URL("https://example.com/x.png"),
        format = "png",
    )
}
