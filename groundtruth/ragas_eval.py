"""Optional LLM-judged quality metrics for saved ablation output."""

import argparse
import json
from pathlib import Path

from groundtruth.benchmark import FIXTURES
from groundtruth.config import settings


def score(input_path: Path, output_path: Path) -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    source = json.loads(input_path.read_text())
    references = {key: statement for key, _, statement, _, _ in FIXTURES}
    samples = [{"user_input": row["question"], "retrieved_contexts": row["retrieved_contexts"],
                "response": row["answer"], "reference": references[row["document_id"]]}
               for row in source["cases"] if row["type"] == "answerable" and row["answer"]]
    if not samples:
        raise ValueError("No answered cases to score")
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for Ragas scoring")
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset,
                      metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), ResponseRelevancy()],
                      llm=LangchainLLMWrapper(ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key, base_url=settings.openai_base_url)),
                      embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url)))
    output_path.write_text(json.dumps({"source": str(input_path), "sample_size": len(samples), "ragas": dict(result)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    score(args.input, args.output)
