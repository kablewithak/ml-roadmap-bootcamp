"""Evaluation harness for RAG Assistant.

Runs predefined queries and measures retrieval quality.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from rag.config import Config
from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# Default evaluation queries
DEFAULT_EVAL_QUERIES = [
    {
        "query": "What is RAG?",
        "expected_nuggets": ["retrieval", "augmented", "generation"],
        "expected_sources": ["glossary_ai_rag.md"],
    },
    {
        "query": "How do I define a function in Python?",
        "expected_nuggets": ["def", "function", "return"],
        "expected_sources": ["python_basics.md"],
    },
    {
        "query": "What are embeddings?",
        "expected_nuggets": ["vector", "semantic", "representation"],
        "expected_sources": ["glossary_ai_rag.md"],
    },
    {
        "query": "What is the remote work policy?",
        "expected_nuggets": ["remote", "days", "week"],
        "expected_sources": ["policies.md"],
    },
    {
        "query": "How do I handle errors in Python?",
        "expected_nuggets": ["try", "except", "exception"],
        "expected_sources": ["python_basics.md"],
    },
    {
        "query": "What is top-k retrieval?",
        "expected_nuggets": ["top", "similar", "documents"],
        "expected_sources": ["glossary_ai_rag.md"],
    },
    {
        "query": "How do I get started with this project?",
        "expected_nuggets": ["install", "ollama", "ingest"],
        "expected_sources": ["project_faq.md"],
    },
    {
        "query": "What is the code review policy?",
        "expected_nuggets": ["pull request", "review", "approval"],
        "expected_sources": ["policies.md"],
    },
]


class EvalHarness:
    """Evaluation harness for measuring retrieval quality."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "kb_docs",
        top_k: int = 4,
        eval_queries: Optional[list[dict]] = None,
    ):
        """Initialize the evaluation harness.

        Args:
            persist_dir: Path to ChromaDB persistence directory.
            collection_name: Name of the ChromaDB collection.
            top_k: Number of documents to retrieve.
            eval_queries: Custom evaluation queries (uses defaults if None).
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.eval_queries = eval_queries or DEFAULT_EVAL_QUERIES

        # Initialize retriever
        self.retriever = DocumentRetriever(
            persist_dir=persist_dir,
            top_k=top_k,
            score_threshold=0.0,  # Don't filter for evaluation
        )

    def run_evaluation(self) -> dict:
        """Run evaluation on all queries.

        Returns:
            Evaluation results dictionary.
        """
        logger.info(f"Running evaluation with {len(self.eval_queries)} queries")
        start_time = time.time()

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "top_k": self.top_k,
                "collection": self.collection_name,
                "num_queries": len(self.eval_queries),
            },
            "queries": [],
            "metrics": {},
        }

        total_hits = 0
        total_nuggets = 0
        total_expected_nuggets = 0
        total_source_hits = 0
        total_expected_sources = 0

        for query_spec in self.eval_queries:
            query_result = self._evaluate_query(query_spec)
            results["queries"].append(query_result)

            # Accumulate metrics
            total_hits += query_result["source_hit"]
            total_nuggets += query_result["nuggets_found"]
            total_expected_nuggets += len(query_spec["expected_nuggets"])
            total_source_hits += query_result["source_hit"]
            total_expected_sources += 1

        # Calculate aggregate metrics
        elapsed = time.time() - start_time
        num_queries = len(self.eval_queries)

        results["metrics"] = {
            "hit_rate": total_hits / num_queries if num_queries > 0 else 0,
            "nugget_recall": (
                total_nuggets / total_expected_nuggets
                if total_expected_nuggets > 0
                else 0
            ),
            "avg_retrieval_time": elapsed / num_queries if num_queries > 0 else 0,
            "total_time": elapsed,
        }

        logger.info(
            f"Evaluation complete: hit_rate={results['metrics']['hit_rate']:.2%}, "
            f"nugget_recall={results['metrics']['nugget_recall']:.2%}"
        )

        return results

    def _evaluate_query(self, query_spec: dict) -> dict:
        """Evaluate a single query.

        Args:
            query_spec: Query specification with expected results.

        Returns:
            Evaluation result for this query.
        """
        query = query_spec["query"]
        expected_nuggets = query_spec["expected_nuggets"]
        expected_sources = query_spec.get("expected_sources", [])

        # Run retrieval
        start_time = time.time()
        results = self.retriever.retrieve(
            query=query,
            collection_name=self.collection_name,
        )
        elapsed = time.time() - start_time

        # Check for nuggets in retrieved content
        combined_content = " ".join(r["content"].lower() for r in results)
        nuggets_found = sum(
            1 for nugget in expected_nuggets if nugget.lower() in combined_content
        )

        # Check for expected sources
        retrieved_sources = [r["metadata"].get("source", "") for r in results]
        source_hit = any(
            expected in retrieved_sources for expected in expected_sources
        )

        # Get top score
        top_score = results[0]["score"] if results else 0

        return {
            "query": query,
            "expected_nuggets": expected_nuggets,
            "expected_sources": expected_sources,
            "nuggets_found": nuggets_found,
            "nuggets_total": len(expected_nuggets),
            "source_hit": 1 if source_hit else 0,
            "top_score": top_score,
            "num_results": len(results),
            "retrieval_time": elapsed,
            "retrieved_sources": retrieved_sources[:self.top_k],
        }

    def generate_report(self, results: dict) -> str:
        """Generate a markdown report from evaluation results.

        Args:
            results: Evaluation results dictionary.

        Returns:
            Markdown formatted report.
        """
        report_lines = [
            "# RAG Assistant Evaluation Report",
            "",
            f"**Date**: {results['timestamp']}",
            f"**Collection**: {results['config']['collection']}",
            f"**Top-K**: {results['config']['top_k']}",
            f"**Queries**: {results['config']['num_queries']}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Hit Rate | {results['metrics']['hit_rate']:.1%} |",
            f"| Nugget Recall | {results['metrics']['nugget_recall']:.1%} |",
            f"| Avg Retrieval Time | {results['metrics']['avg_retrieval_time']*1000:.1f}ms |",
            f"| Total Time | {results['metrics']['total_time']:.2f}s |",
            "",
            "## Query Results",
            "",
        ]

        for i, qr in enumerate(results["queries"], 1):
            status = "✓" if qr["source_hit"] else "✗"
            nugget_ratio = f"{qr['nuggets_found']}/{qr['nuggets_total']}"

            report_lines.extend([
                f"### {i}. {qr['query']}",
                "",
                f"- **Source Hit**: {status}",
                f"- **Nuggets**: {nugget_ratio}",
                f"- **Top Score**: {qr['top_score']:.3f}",
                f"- **Time**: {qr['retrieval_time']*1000:.1f}ms",
                f"- **Retrieved**: {', '.join(qr['retrieved_sources'][:3])}",
                "",
            ])

        # Add qualitative notes section
        report_lines.extend([
            "## Notes",
            "",
            "_Add qualitative observations here:_",
            "",
            "- [ ] Are the top results relevant?",
            "- [ ] Are there any surprising misses?",
            "- [ ] Should chunk size be adjusted?",
            "- [ ] Should top-k be increased/decreased?",
            "",
        ])

        return "\n".join(report_lines)

    def save_report(self, results: dict, output_path: Optional[Path] = None) -> Path:
        """Save evaluation report to file.

        Args:
            results: Evaluation results dictionary.
            output_path: Output file path (defaults to eval/report.md).

        Returns:
            Path to saved report.
        """
        if output_path is None:
            eval_dir = Config.BASE_DIR / "eval"
            eval_dir.mkdir(exist_ok=True)
            output_path = eval_dir / "report.md"

        report = self.generate_report(results)
        output_path.write_text(report)

        logger.info(f"Report saved to {output_path}")
        return output_path


def run_evaluation(
    top_k: int = 4,
    output_path: Optional[Path] = None,
) -> dict:
    """Convenience function to run evaluation.

    Args:
        top_k: Number of documents to retrieve.
        output_path: Output file path for report.

    Returns:
        Evaluation results.
    """
    harness = EvalHarness(top_k=top_k)
    results = harness.run_evaluation()
    harness.save_report(results, output_path)
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run evaluation
    print("Running RAG evaluation harness...")
    results = run_evaluation()

    # Print summary
    print("\nEvaluation Summary:")
    print(f"  Hit Rate: {results['metrics']['hit_rate']:.1%}")
    print(f"  Nugget Recall: {results['metrics']['nugget_recall']:.1%}")
    print(f"  Total Time: {results['metrics']['total_time']:.2f}s")
    print(f"\nReport saved to eval/report.md")
