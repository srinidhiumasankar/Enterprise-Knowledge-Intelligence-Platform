# tests/test_self_query_mock.py
# -----------------------------
# Unit tests using mocks to verify normalization helper and execution pipeline.

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.structured_query import Comparison, Operation, StructuredQuery, Comparator, Operator
from langchain_core.documents import Document

from app.services.langchain.self_query import (
    parse_filter,
    normalize_structured_query,
    ChromaSelfQueryRetriever
)

class TestSelfQueryNormalization(unittest.TestCase):

    def test_parse_filter_none(self):
        self.assertIsNone(parse_filter(None))

    def test_parse_filter_ast(self):
        # If it is already Comparison or Operation, should return as is
        c = Comparison(comparator=Comparator.GT, attribute="year", value=2022)
        self.assertEqual(parse_filter(c), c)

        op = Operation(operator=Operator.AND, arguments=[c])
        self.assertEqual(parse_filter(op), op)

    def test_parse_filter_dict_comparison(self):
        # Dictionary Comparison parsing
        filter_dict = {"comparator": "gt", "attribute": "year", "value": 2022}
        parsed = parse_filter(filter_dict)
        self.assertIsInstance(parsed, Comparison)
        self.assertEqual(parsed.comparator, Comparator.GT)
        self.assertEqual(parsed.attribute, "year")
        self.assertEqual(parsed.value, 2022)

    def test_parse_filter_dict_operation(self):
        # Dictionary Operation parsing
        filter_dict = {
            "operator": "and",
            "arguments": [
                {"comparator": "gt", "attribute": "year", "value": 2022},
                {"comparator": "eq", "attribute": "department", "value": "Finance"}
            ]
        }
        parsed = parse_filter(filter_dict)
        self.assertIsInstance(parsed, Operation)
        self.assertEqual(parsed.operator, Operator.AND)
        self.assertEqual(len(parsed.arguments), 2)
        self.assertIsInstance(parsed.arguments[0], Comparison)
        self.assertEqual(parsed.arguments[0].comparator, Comparator.GT)
        self.assertIsInstance(parsed.arguments[1], Comparison)
        self.assertEqual(parsed.arguments[1].comparator, Comparator.EQ)

    def test_normalize_structured_query_none(self):
        normalized = normalize_structured_query(None)
        self.assertIsInstance(normalized, StructuredQuery)
        self.assertEqual(normalized.query, "")
        self.assertIsNone(normalized.filter)

    def test_normalize_structured_query_ast(self):
        c = Comparison(comparator=Comparator.GT, attribute="year", value=2022)
        sq = StructuredQuery(query="Finance report", filter=c, limit=5)
        normalized = normalize_structured_query(sq)
        self.assertIsInstance(normalized, StructuredQuery)
        self.assertEqual(normalized.query, "Finance report")
        self.assertEqual(normalized.filter, c)
        self.assertEqual(normalized.limit, 5)

    def test_normalize_structured_query_dict_flat(self):
        sq_dict = {
            "query": "Finance report",
            "filter": {"comparator": "gt", "attribute": "year", "value": 2022},
            "limit": 5
        }
        normalized = normalize_structured_query(sq_dict)
        self.assertIsInstance(normalized, StructuredQuery)
        self.assertEqual(normalized.query, "Finance report")
        self.assertIsInstance(normalized.filter, Comparison)
        self.assertEqual(normalized.filter.comparator, Comparator.GT)
        self.assertEqual(normalized.limit, 5)

    def test_normalize_structured_query_dict_nested(self):
        # Mocking output returned by LLMChain which puts parsed query into 'text'
        raw_chain_output = {
            "query": "Finance report after 2022",
            "text": {
                "query": "Finance report",
                "filter": {"comparator": "gt", "attribute": "year", "value": 2022},
                "limit": 3
            }
        }
        normalized = normalize_structured_query(raw_chain_output)
        self.assertIsInstance(normalized, StructuredQuery)
        self.assertEqual(normalized.query, "Finance report")
        self.assertIsInstance(normalized.filter, Comparison)
        self.assertEqual(normalized.filter.comparator, Comparator.GT)
        self.assertEqual(normalized.limit, 3)

    def test_normalize_structured_query_dict_nested_ast(self):
        c = Comparison(comparator=Comparator.GT, attribute="year", value=2022)
        sq = StructuredQuery(query="Finance report", filter=c, limit=3)
        raw_chain_output = {
            "query": "Finance report after 2022",
            "text": sq
        }
        normalized = normalize_structured_query(raw_chain_output)
        self.assertIsInstance(normalized, StructuredQuery)
        self.assertEqual(normalized.query, "Finance report")
        self.assertEqual(normalized.filter, c)
        self.assertEqual(normalized.limit, 3)


class TestChromaSelfQueryRetriever(unittest.TestCase):

    @patch("app.services.langchain.self_query.load_query_constructor_chain")
    @patch("app.services.langchain.hybrid_retriever.get_hybrid_retriever")
    @patch("app.services.langchain.parent_retriever.ParentRetriever")
    @patch("app.services.langchain.compression.CompressionRetriever")
    @patch("app.services.langchain.multi_query.get_multi_query_retriever")
    @patch("app.services.langchain.ensemble.EnsembleRetriever")
    def test_retriever_pipeline_execution(
        self,
        mock_ensemble_retriever,
        mock_get_multi_query_retriever,
        mock_compression_retriever,
        mock_parent_retriever,
        mock_get_hybrid_retriever,
        mock_load_query_constructor_chain
    ):
        # 1. Setup mocks
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_load_query_constructor_chain.return_value = mock_chain

        # Mock query constructor invoke returning a dictionary (like current LangChain output)
        mock_chain.invoke.return_value = {
            "query": "Finance reports after 2022",
            "text": {
                "query": "Finance reports",
                "filter": {"comparator": "gt", "attribute": "year", "value": 2022}
            }
        }

        # Mock downstream hybrid, parent, compression, and multi-query retrievers
        mock_hybrid = MagicMock()
        mock_get_hybrid_retriever.return_value = mock_hybrid

        mock_parent_instance = MagicMock()
        mock_parent_retriever.return_value = mock_parent_instance

        mock_comp_instance = MagicMock()
        mock_compression_retriever.return_value = mock_comp_instance

        mock_mq = MagicMock()
        mock_get_multi_query_retriever.return_value = mock_mq
        
        mock_ensemble_instance = MagicMock()
        mock_ensemble_retriever.return_value = mock_ensemble_instance
        
        expected_docs = [Document(page_content="Mock doc 1", metadata={"year": 2024})]
        mock_ensemble_instance.invoke.return_value = expected_docs

        # 2. Instantiate and run ChromaSelfQueryRetriever
        retriever = ChromaSelfQueryRetriever(
            llm=mock_llm,
            owner_id=999,
            top_k=2
        )
        
        # Invoke retriever
        results = retriever.invoke("Finance reports after 2022")

        # 3. Assertions
        mock_chain.invoke.assert_called_once_with({"query": "Finance reports after 2022"})
        mock_get_hybrid_retriever.assert_called_once()
        mock_ensemble_instance.invoke.assert_called_once_with("Finance reports")
        
        self.assertEqual(results, expected_docs)
        self.assertEqual(mock_hybrid.where_override, {
            "$and": [
                {"owner_id": 999},
                {"year": {"$gt": 2022}}
            ]
        })


if __name__ == "__main__":
    unittest.main()
