package ai.koog.agents.optimization.core

import ai.koog.agents.optimization.utils.messages.toolResults
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.ResponseMetaInfo
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.time.Clock

internal class DemonstrationRendererTest {

    private val requestMeta = RequestMetaInfo(Clock.System.now())
    private val responseMeta = ResponseMetaInfo(Clock.System.now())

    private fun user(content: String) = Message.User(content, requestMeta)

    private fun call(id: String, tool: String = "someTool") =
        MessagePart.Tool.Call(id = id, tool = tool, args = "{}")

    private fun assistantCalls(vararg calls: MessagePart.Tool.Call) =
        Message.Assistant(parts = calls.toList(), metaInfo = responseMeta)

    private fun toolResult(id: String, tool: String = "someTool") =
        Message.User(
            parts = listOf(MessagePart.Tool.Result(id = id, tool = tool, output = "output")),
            metaInfo = requestMeta,
        )

    private fun renderFullTrace(messages: List<Message>): List<Message> =
        DemonstrationRenderer.renderAsMessages(
            listOf(Demonstration("input", "output", intermediateMessages = messages)),
            DemonstrationFormat.FULL_TRACE,
        )

    private fun assertSyntheticResultFor(call: MessagePart.Tool.Call, message: Message) {
        val result = message.toolResults().single { it.id == call.id }
        assertEquals(call.tool, result.tool)
        assertEquals(DemonstrationRenderer.SYNTHETIC_TOOL_RESULT, result.output)
    }

    @Test
    fun testTrailingUnansweredToolCallGetsSyntheticResult() {
        val trailingCall = call("call_2", tool = "commitFinalAnswer")
        val trace = listOf(
            user("question"),
            assistantCalls(call("call_1")),
            toolResult("call_1"),
            assistantCalls(trailingCall),
        )

        val rendered = renderFullTrace(trace)

        assertEquals(5, rendered.size)
        assertEquals(trace, rendered.take(4))
        assertSyntheticResultFor(trailingCall, rendered[4])
    }

    @Test
    fun testAnsweredToolCallsAreLeftUntouched() {
        val trace = listOf(
            user("question"),
            assistantCalls(call("call_1")),
            toolResult("call_1"),
        )

        assertEquals(trace, renderFullTrace(trace))
    }

    @Test
    fun testUnansweredToolCallMidTraceIsClosedBeforeNextMessage() {
        val danglingCall = call("call_1")
        val trace = listOf(
            user("question"),
            assistantCalls(danglingCall),
            user("follow-up"),
        )

        val rendered = renderFullTrace(trace)

        assertEquals(4, rendered.size)
        assertSyntheticResultFor(danglingCall, rendered[2])
        assertEquals(user("follow-up"), rendered[3])
    }

    @Test
    fun testParallelToolCallBlockClosesOnlyUnansweredCalls() {
        val answeredCall = call("call_1")
        val danglingCall = call("call_2")
        val trace = listOf(
            user("question"),
            assistantCalls(answeredCall, danglingCall),
            toolResult("call_1"),
        )

        val rendered = renderFullTrace(trace)

        // The synthetic result joins the message that answered the rest of the batch — Anthropic
        // rejects a `tool_result` that is not in the single user turn right after the `tool_use`.
        assertEquals(3, rendered.size)
        assertEquals(trace.take(2), rendered.take(2))
        assertEquals(listOf("call_1", "call_2"), rendered[2].toolResults().map { it.id })
        assertSyntheticResultFor(danglingCall, rendered[2])
    }

    @Test
    fun testCompactDemosAreUnaffected() {
        val rendered = DemonstrationRenderer.renderAsMessages(
            listOf(Demonstration("some input", "some output")),
            DemonstrationFormat.FULL_TRACE,
        )

        assertEquals(2, rendered.size)
        assertEquals("some input", assertIs<Message.User>(rendered[0]).textContent())
        assertEquals("some output", assertIs<Message.Assistant>(rendered[1]).textContent())
    }
}
