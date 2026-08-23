"""cost_tracking.py's pure functions, plus an end-to-end check that
observability.usage_tracking_config()'s UsageMetadataCallbackHandler
actually captures usage through a real rag_pipeline chain.

UsageMetadataCallbackHandler only aggregates a call if the returned
AIMessage carries both `usage_metadata` *and* a `model_name` in its
`response_metadata` -- LangChain core's own BaseChatModel.generate()
machinery merges a chat model's returned ChatResult.llm_output into each
message's response_metadata, which is how ChatGroq/ChatOpenAI populate
`model_name` in practice (confirmed against langchain_groq's source:
ChatGroq._create_chat_result sets usage_metadata directly on the message
and puts model_name in ChatResult.llm_output). FakeListChatModel/
FakeMessagesListChatModel bypass that machinery and don't reproduce this,
so the fake model below implements _generate() the same way ChatGroq does,
to test the real merging path rather than a shortcut that would pass
without proving anything.
"""
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cost_tracking import estimate_cost_usd, latency_summary, percentile, timed, total_tokens_and_cost
from observability import usage_tracking_config
from rag_pipeline import answer, build_hybrid_retriever, build_qa_chain, build_vector_store, load_and_chunk


class _FakeGroqLikeChatModel(BaseChatModel):
    """Same ChatResult/llm_output shape as ChatGroq._create_chat_result."""

    model_name: str = "mixtral-8x7b-32768"
    canned_answer: str = "fake answer"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        message = AIMessage(content=self.canned_answer)
        message.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        generation = ChatGeneration(message=message, generation_info={"finish_reason": "stop"})
        return ChatResult(generations=[generation], llm_output={"token_usage": {}, "model_name": self.model_name})

    @property
    def _llm_type(self) -> str:
        return "fake-groq-like"


def test_percentile_and_latency_summary():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 50) is None
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["mean_seconds"] == 2.5
    assert summary["max_seconds"] == 4.0


def test_estimate_cost_usd_known_and_unknown_model():
    cost = estimate_cost_usd("gpt-3.5-turbo", 1_000_000, 1_000_000)
    assert abs(cost - 2.0) < 1e-9
    assert estimate_cost_usd("not-a-real-model", 1000, 1000) is None


def test_total_tokens_and_cost_mixed_known_unknown_models():
    usage = {
        "gpt-3.5-turbo": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        "some-unpriced-model": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
    }
    result = total_tokens_and_cost(usage)
    assert result["total_tokens"] == 3000
    # only the priced model contributes to cost_usd, but it's still non-None
    assert result["cost_usd"] is not None


def test_usage_handler_captures_real_generate_pipeline_usage(fixture_pdf_paths):
    """The actual regression this suite exists to catch: a naive fake model
    (bare AIMessage, no ChatResult.llm_output) silently produces an empty
    usage_metadata dict -- this asserts the real BaseChatModel-contract path
    (what ChatGroq/ChatOpenAI actually do) is captured correctly."""
    chunks = load_and_chunk(fixture_pdf_paths)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(vector_store, chunks)
    fake_llm = _FakeGroqLikeChatModel(canned_answer="The stipend is $750.")
    chain = build_qa_chain(retriever, llm=fake_llm)

    _, usage_handler = usage_tracking_config()
    with timed() as t:
        result = answer("What is the equipment stipend amount?", chain, usage_handler=usage_handler)

    assert result["answer"] == "The stipend is $750."
    assert t.elapsed_seconds is not None and t.elapsed_seconds >= 0
    assert usage_handler.usage_metadata == {
        "mixtral-8x7b-32768": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    }

    tc = total_tokens_and_cost(usage_handler.usage_metadata)
    assert tc["total_tokens"] == 120
    assert tc["cost_usd"] is not None
