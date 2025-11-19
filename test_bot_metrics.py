import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from bot import ConstructionMaterialsBot
from src.testing.evaluator import ResponseEvaluator
from src.testing.query_generator import QueryGenerator
from src.testing.simulated_buyer import SimulatedBuyer


class BotTester:
    """Main testing class for bot evaluation."""
    
    def __init__(
        self,
        use_in_memory: bool = False,
        evaluator_model: str = "mistral-small",
        buyer_model: str = "mistral-small"
    ):
        """
        Initialize the bot tester.
        
        Args:
            use_in_memory: Whether to use in-memory Qdrant
            evaluator_model: Model for evaluation
            buyer_model: Model for simulated buyer
        """
        print("=" * 80)
        print("Initializing Bot Tester")
        print("=" * 80)
        
        # Initialize bot
        print("\n1. Initializing Construction Materials Bot...")
        self.bot = ConstructionMaterialsBot(use_in_memory=use_in_memory)
        
        # Initialize evaluator
        print("\n2. Initializing Response Evaluator...")
        self.evaluator = ResponseEvaluator(model=evaluator_model)
        
        # Initialize query generator
        print("\n3. Initializing Query Generator...")
        self.query_generator = QueryGenerator(model=evaluator_model)
        
        # Initialize simulated buyer
        print("\n4. Initializing Simulated Buyer...")
        self.simulated_buyer = SimulatedBuyer(model=buyer_model)
        
        print("\n" + "=" * 80)
        print("Initialization Complete!")
        print("=" * 80)
    
    def test_single_query(
        self,
        query: str,
        session_id: str = "test_session"
    ) -> Dict[str, Any]:
        """
        Test a single query and evaluate the response.
        
        Args:
            query: Test query
            session_id: Session identifier
            
        Returns:
            Dictionary with test results and evaluation
        """
        print(f"\n{'='*80}")
        print(f"Testing Query: {query}")
        print(f"{'='*80}")
        
        # Get bot response
        print("\nProcessing query with bot...")
        bot_response = self.bot.process_query(query, session_id=session_id)
        
        print(f"\nBot Response:")
        print(f"  Message: {bot_response.message[:200]}...")
        print(f"  Query Type: {bot_response.query_type}")
        print(f"  Needs Clarification: {bot_response.needs_clarification}")
        
        # Evaluate response
        print("\nEvaluating response...")
        evaluation = self.evaluator.evaluate(
            query=query,
            response=bot_response.message,
            query_type=bot_response.query_type,
            needs_clarification=bot_response.needs_clarification,
            extracted_specs=bot_response.extracted_specs.dict() if bot_response.extracted_specs else None
        )
        
        print(f"\nEvaluation Results:")
        print(f"  Overall Score: {evaluation['score']:.3f}")
        print(f"  Relevance: {evaluation.get('relevance', 0):.3f}")
        print(f"  Completeness: {evaluation.get('completeness', 0):.3f}")
        print(f"  Accuracy: {evaluation.get('accuracy', 0):.3f}")
        print(f"  Clarity: {evaluation.get('clarity', 0):.3f}")
        if 'reasoning' in evaluation:
            print(f"  Reasoning: {evaluation['reasoning'][:200]}...")
        
        return {
            "query": query,
            "bot_response": {
                "message": bot_response.message,
                "query_type": bot_response.query_type,
                "needs_clarification": bot_response.needs_clarification,
                "extracted_specs": bot_response.extracted_specs.dict() if bot_response.extracted_specs else None
            },
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_with_simulated_buyer(
        self,
        initial_query: str,
        max_turns: int = 5,
        session_id: str = "simulated_session"
    ) -> Dict[str, Any]:
        """
        Test bot with a simulated buyer in a multi-turn conversation.
        
        Args:
            initial_query: Initial query from buyer
            max_turns: Maximum number of conversation turns
            session_id: Session identifier
            
        Returns:
            Dictionary with conversation and evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Simulated Buyer Conversation")
        print(f"Initial Query: {initial_query}")
        print(f"{'='*80}")
        
        conversation = []
        self.simulated_buyer.reset()
        
        # Start conversation
        buyer_query = self.simulated_buyer.start_conversation(initial_query)
        conversation.append({"role": "buyer", "message": buyer_query})
        
        print(f"\nBuyer: {buyer_query}")
        
        for turn in range(max_turns):
            # Get bot response
            bot_response = self.bot.process_query(buyer_query, session_id=session_id)
            conversation.append({
                "role": "bot",
                "message": bot_response.message,
                "query_type": bot_response.query_type,
                "needs_clarification": bot_response.needs_clarification
            })
            
            print(f"\nBot: {bot_response.message[:200]}...")
            
            # Evaluate this turn
            evaluation = self.evaluator.evaluate(
                query=buyer_query,
                response=bot_response.message,
                query_type=bot_response.query_type,
                needs_clarification=bot_response.needs_clarification
            )
            conversation[-1]["evaluation"] = evaluation
            
            print(f"  Score: {evaluation['score']:.3f}")
            
            # Check if conversation should end
            if not bot_response.needs_clarification and bot_response.query_type == "order_specification":
                print("\n✓ Order completed, conversation ending.")
                break
            
            # Generate buyer response
            buyer_query = self.simulated_buyer.respond(bot_response.message)
            conversation.append({"role": "buyer", "message": buyer_query})
            
            print(f"\nBuyer: {buyer_query}")
        
        # Calculate average score
        evaluations = [turn["evaluation"] for turn in conversation if "evaluation" in turn]
        avg_score = sum(e["score"] for e in evaluations) / len(evaluations) if evaluations else 0.0
        
        print(f"\n{'='*80}")
        print(f"Conversation Complete")
        print(f"Average Score: {avg_score:.3f}")
        print(f"{'='*80}")
        
        return {
            "initial_query": initial_query,
            "conversation": conversation,
            "average_score": avg_score,
            "turns": len([t for t in conversation if t["role"] == "bot"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_batch_test(
        self,
        queries: List[Dict[str, Any]],
        use_simulated_buyer: bool = False
    ) -> Dict[str, Any]:
        """
        Run batch testing on multiple queries.
        
        Args:
            queries: List of query dictionaries
            use_simulated_buyer: Whether to use simulated buyer for multi-turn conversations
            
        Returns:
            Dictionary with batch test results
        """
        print(f"\n{'='*80}")
        print(f"Running Batch Test")
        print(f"Number of queries: {len(queries)}")
        print(f"Use simulated buyer: {use_simulated_buyer}")
        print(f"{'='*80}")
        
        results = []
        
        for i, query_data in enumerate(queries, 1):
            query = query_data.get("query", query_data) if isinstance(query_data, dict) else query_data
            
            print(f"\n[{i}/{len(queries)}] Processing query...")
            
            if use_simulated_buyer:
                result = self.test_with_simulated_buyer(
                    initial_query=query,
                    max_turns=5,
                    session_id=f"batch_session_{i}"
                )
            else:
                result = self.test_single_query(
                    query=query,
                    session_id=f"batch_session_{i}"
                )
            
            results.append(result)
        
        # Calculate aggregate metrics
        all_scores = []
        for result in results:
            if "evaluation" in result:
                all_scores.append(result["evaluation"]["score"])
            elif "average_score" in result:
                all_scores.append(result["average_score"])
        
        aggregate_metrics = {
            "total_queries": len(queries),
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "min_score": min(all_scores) if all_scores else 0.0,
            "max_score": max(all_scores) if all_scores else 0.0,
            "scores_above_0.7": sum(1 for s in all_scores if s >= 0.7),
            "scores_above_0.5": sum(1 for s in all_scores if s >= 0.5),
            "scores_below_0.5": sum(1 for s in all_scores if s < 0.5)
        }
        
        print(f"\n{'='*80}")
        print(f"Batch Test Complete")
        print(f"{'='*80}")
        print(f"\nAggregate Metrics:")
        print(f"  Total Queries: {aggregate_metrics['total_queries']}")
        print(f"  Average Score: {aggregate_metrics['average_score']:.3f}")
        print(f"  Min Score: {aggregate_metrics['min_score']:.3f}")
        print(f"  Max Score: {aggregate_metrics['max_score']:.3f}")
        print(f"  Scores ≥ 0.7: {aggregate_metrics['scores_above_0.7']}")
        print(f"  Scores ≥ 0.5: {aggregate_metrics['scores_above_0.5']}")
        print(f"  Scores < 0.5: {aggregate_metrics['scores_below_0.5']}")
        
        return {
            "results": results,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save test results to a JSON file.
        
        Args:
            results: Test results dictionary
            filename: Output filename (default: test_results_<timestamp>.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        output_path = Path("test_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Construction Materials Bot with metrics")
    parser.add_argument(
        "--use-in-memory",
        action="store_true",
        help="Use in-memory Qdrant (for testing)"
    )
    parser.add_argument(
        "--generate-queries",
        type=int,
        default=0,
        help="Generate N test queries automatically"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Specific queries to test"
    )
    parser.add_argument(
        "--simulated-buyer",
        action="store_true",
        help="Use simulated buyer for multi-turn conversations"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = BotTester(use_in_memory=args.use_in_memory)
    
    # Prepare queries
    queries = []
    
    if args.generate_queries > 0:
        print(f"\nGenerating {args.generate_queries} test queries...")
        generated_queries = tester.query_generator.generate_queries(args.generate_queries)
        queries.extend(generated_queries)
    
    if args.queries:
        queries.extend([{"query": q, "type": "unknown"} for q in args.queries])
    
    if not queries:
        # Default test queries
        queries = [
            {"query": "Какие характеристики у бетона М300?", "type": "informational"},
            {"query": "Хочу заказать бетон М300", "type": "order_specification"},
            {"query": "Нужен песок для строительства", "type": "order_specification"}
        ]
    
    # Run tests
    if args.simulated_buyer:
        # Test with simulated buyer
        results = []
        for query_data in queries:
            query = query_data.get("query", query_data) if isinstance(query_data, dict) else query_data
            result = tester.test_with_simulated_buyer(query, max_turns=5)
            results.append(result)
        
        batch_results = {
            "results": results,
            "aggregate_metrics": {
                "average_score": sum(r["average_score"] for r in results) / len(results) if results else 0.0
            }
        }
    else:
        # Run batch test
        batch_results = tester.run_batch_test(queries, use_simulated_buyer=False)
    
    # Save results if requested
    if args.save_results:
        tester.save_results(batch_results, args.save_results)
    else:
        tester.save_results(batch_results)


if __name__ == "__main__":
    main()

