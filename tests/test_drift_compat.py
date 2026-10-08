import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from automotive_graphrag.drift_compat import (
    SamplingParameterFilter,
    _filter_engine_model,
    run_drift_search,
)


class FakeCompletion:
    tokenizer = object()

    def __init__(self):
        self.calls = []

    async def completion_async(self, **kwargs):
        self.calls.append(kwargs)
        return "async response"

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return "sync response"


class DriftCompatibilityTests(unittest.TestCase):
    def test_sampling_parameters_are_removed_only_at_completion_boundary(self):
        completion = FakeCompletion()
        adapter = SamplingParameterFilter(completion)

        response = asyncio.run(adapter.completion_async(
            messages="prompt", temperature=0, top_p=1, n=1, reasoning_effort="medium"
        ))

        self.assertEqual(response, "async response")
        self.assertEqual(
            completion.calls,
            [{"messages": "prompt", "n": 1, "reasoning_effort": "medium"}],
        )
        self.assertIs(adapter.tokenizer, completion.tokenizer)

    def test_gpt6_luna_engine_model_references_use_adapter(self):
        completion = FakeCompletion()
        engine = SimpleNamespace(
            model=completion,
            context_builder=SimpleNamespace(model=completion),
            primer=SimpleNamespace(chat_model=completion),
            local_search=SimpleNamespace(model=completion),
        )
        config = SimpleNamespace(
            drift_search=SimpleNamespace(completion_model_id="drift"),
            get_completion_model_config=lambda _model_id: SimpleNamespace(model="gpt-6-luna"),
        )

        adapted = _filter_engine_model(engine, config)

        self.assertIs(adapted.model, adapted.context_builder.model)
        self.assertIs(adapted.model, adapted.primer.chat_model)
        self.assertIs(adapted.model, adapted.local_search.model)
        self.assertIsInstance(adapted.model, SamplingParameterFilter)

    def test_other_models_are_not_wrapped(self):
        completion = FakeCompletion()
        engine = SimpleNamespace(model=completion)
        config = SimpleNamespace(
            drift_search=SimpleNamespace(completion_model_id="drift"),
            get_completion_model_config=lambda _model_id: SimpleNamespace(model="gpt-4o-mini"),
        )

        self.assertIs(_filter_engine_model(engine, config).model, completion)

    def test_drift_entrypoint_uses_scoped_luna_adapter(self):
        from graphrag.api import query as query_api
        from graphrag.cli import query as cli_query

        original_factory = query_api.get_drift_search_engine
        completion = FakeCompletion()
        config = SimpleNamespace(
            drift_search=SimpleNamespace(completion_model_id="drift"),
            get_completion_model_config=lambda _model_id: SimpleNamespace(model="gpt-6-luna"),
        )
        engine = SimpleNamespace(
            model=completion,
            context_builder=SimpleNamespace(model=completion),
            primer=SimpleNamespace(chat_model=completion),
            local_search=SimpleNamespace(model=completion),
        )

        def fake_factory(*, config):
            return engine

        async def fake_completion_call():
            created = query_api.get_drift_search_engine(config=config)
            await created.model.completion_async(
                messages="prompt", temperature=0, top_p=1, max_completion_tokens=50
            )

        def fake_cli_run(**_kwargs):
            asyncio.run(fake_completion_call())
            return "Luna response", {"sources": []}

        with (
            patch.object(query_api, "get_drift_search_engine", fake_factory),
            patch.object(cli_query, "run_drift_search", fake_cli_run),
        ):
            answer, context = run_drift_search(root_dir="unused", query="question")

        self.assertEqual(answer, "Luna response")
        self.assertEqual(context, {"sources": []})
        self.assertEqual(
            completion.calls,
            [{"messages": "prompt", "max_completion_tokens": 50}],
        )
        self.assertIs(query_api.get_drift_search_engine, original_factory)


if __name__ == "__main__":
    unittest.main()
