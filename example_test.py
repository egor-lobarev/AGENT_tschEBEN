"""
Simple example of using the testing framework.
"""

import os
from dotenv import load_dotenv
from bot import ConstructionMaterialsBot
from src.testing.evaluator import ResponseEvaluator
from src.testing.query_generator import QueryGenerator
from src.testing.simulated_buyer import SimulatedBuyer

load_dotenv()


def example_single_evaluation():
    """Example: Evaluate a single query-response pair."""
    print("=" * 80)
    print("Example 1: Single Query Evaluation")
    print("=" * 80)
    
    # Initialize bot and evaluator
    # Use persistent storage to reuse existing vector database
    bot = ConstructionMaterialsBot(use_in_memory=False)
    # Use smaller model to avoid rate limits (mistral-small or mistral-tiny)
    evaluator = ResponseEvaluator(model="mistral-small")
    
    # Test query
    query = "Какие характеристики у бетона М300?"
    print(f"\nQuery: {query}")
    
    # Get bot response
    response = bot.process_query(query)
    print(f"\nBot Response: {response.message[:200]}...")
    
    # Evaluate
    evaluation = evaluator.evaluate(
        query=query,
        response=response.message,
        query_type=response.query_type,
        needs_clarification=response.needs_clarification
    )
    
    print(f"\nEvaluation:")
    print(f"  Score: {evaluation['score']:.3f}")
    print(f"  Relevance: {evaluation.get('relevance', 0):.3f}")
    print(f"  Completeness: {evaluation.get('completeness', 0):.3f}")
    print(f"  Accuracy: {evaluation.get('accuracy', 0):.3f}")
    print(f"  Clarity: {evaluation.get('clarity', 0):.3f}")


def example_simulated_buyer():
    """Example: Simulated buyer conversation."""
    print("\n" + "=" * 80)
    print("Example 2: Simulated Buyer Conversation")
    print("=" * 80)
    
    # Initialize bot and simulated buyer
    # Use persistent storage to reuse existing vector database
    bot = ConstructionMaterialsBot(use_in_memory=False)
    # Use smaller models to avoid rate limits
    buyer = SimulatedBuyer(model="mistral-small")
    evaluator = ResponseEvaluator(model="mistral-small")
    
    # Start conversation
    initial_query = "Хочу заказать бетон М300"
    buyer_query = buyer.start_conversation(initial_query)
    
    print(f"\nBuyer: {buyer_query}")
    
    # Conversation loop
    for turn in range(3):
        # Bot responds
        bot_response = bot.process_query(buyer_query, session_id="example_session")
        print(f"\nBot: {bot_response.message[:150]}...")
        
        # Evaluate
        evaluation = evaluator.evaluate(
            query=buyer_query,
            response=bot_response.message,
            query_type=bot_response.query_type,
            needs_clarification=bot_response.needs_clarification
        )
        print(f"  Score: {evaluation['score']:.3f}")
        
        # Buyer responds
        if not bot_response.needs_clarification and bot_response.query_type == "order_specification":
            print("\n✓ Order completed!")
            break
        
        buyer_query = buyer.respond(bot_response.message)
        print(f"\nBuyer: {buyer_query}")


def example_query_generation():
    """Example: Generate and test queries."""
    print("\n" + "=" * 80)
    print("Example 3: Query Generation and Testing")
    print("=" * 80)
    
    # Initialize components
    # Use persistent storage to reuse existing vector database
    bot = ConstructionMaterialsBot(use_in_memory=False)
    # Use smaller models to avoid rate limits
    generator = QueryGenerator(model="mistral-small")
    evaluator = ResponseEvaluator(model="mistral-small")
    
    # Generate queries
    print("\nGenerating test queries...")
    queries = generator.generate_queries(count=5)
    
    print(f"\nGenerated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q['query']} ({q.get('type', 'unknown')})")
    
    # Test each query
    print("\n" + "-" * 80)
    print("Testing queries...")
    print("-" * 80)
    
    scores = []
    for query_data in queries:
        query = query_data['query']
        response = bot.process_query(query)
        
        evaluation = evaluator.evaluate(
            query=query,
            response=response.message,
            query_type=response.query_type,
            needs_clarification=response.needs_clarification
        )
        
        scores.append(evaluation['score'])
        print(f"\nQuery: {query}")
        print(f"  Score: {evaluation['score']:.3f}")
    
    print(f"\n{'='*80}")
    print(f"Average Score: {sum(scores)/len(scores):.3f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Check API key
    if not os.getenv("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not set in environment")
        print("Please set it in your .env file or environment variables")
        exit(1)
    
    # Run examples
    try:
        example_single_evaluation()
        example_simulated_buyer()
        example_query_generation()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

