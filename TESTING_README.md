# Bot Testing Framework

Comprehensive testing framework for the Construction Materials Bot with automated evaluation, query generation, and simulated buyer interactions.

## Features

- **Automated Evaluation**: Uses a larger Mistral model (mistral-large-latest) to score bot responses on a 0-1 scale
- **Query Generation**: Automatically generates realistic test queries
- **Simulated Buyer**: Neural network-based buyer that can have multi-turn conversations with the bot
- **Metrics Collection**: Comprehensive metrics including relevance, completeness, accuracy, and clarity scores

## Components

### 1. Response Evaluator (`src/testing/evaluator.py`)

Evaluates bot responses using a larger Mistral model. Scores responses on:
- **Relevance** (0.0-0.3): Does the response address the query?
- **Completeness** (0.0-0.3): Is the response complete and informative?
- **Accuracy** (0.0-0.2): Is the information correct?
- **Clarity** (0.0-0.2): Is the response clear and well-structured?

**Total Score**: 0.0 to 1.0

### 2. Query Generator (`src/testing/query_generator.py`)

Generates realistic test queries that customers might ask. Can generate:
- Informational queries (asking about material properties)
- Order specification queries (wanting to place orders)

### 3. Simulated Buyer (`src/testing/simulated_buyer.py`)

A neural network-based agent that simulates a buyer interacting with the bot. Can:
- Start conversations with initial queries
- Respond naturally to bot messages
- Maintain conversation context
- Have multi-turn dialogues

### 4. Bot Tester (`test_bot_metrics.py`)

Main testing script that orchestrates all components.

## Usage

### Basic Testing

Test a single query:

```bash
python test_bot_metrics.py --queries "Какие характеристики у бетона М300?"
```

### Generate and Test Queries

Generate 10 test queries and evaluate them:

```bash
python test_bot_metrics.py --generate-queries 10
```

### Test with Simulated Buyer

Test with a simulated buyer for multi-turn conversations:

```bash
python test_bot_metrics.py --queries "Хочу заказать бетон М300" --simulated-buyer
```

### Save Results

Save test results to a JSON file:

```bash
python test_bot_metrics.py --generate-queries 20 --save-results results.json
```

### Using In-Memory Mode

For faster testing (data lost on restart):

```bash
python test_bot_metrics.py --generate-queries 10 --use-in-memory
```

## Programmatic Usage

### Single Query Test

```python
from bot import ConstructionMaterialsBot
from src.testing.evaluator import ResponseEvaluator

# Initialize bot and evaluator
bot = ConstructionMaterialsBot()
evaluator = ResponseEvaluator()

# Test a query
query = "Какие характеристики у бетона М300?"
response = bot.process_query(query)

# Evaluate response
evaluation = evaluator.evaluate(
    query=query,
    response=response.message,
    query_type=response.query_type,
    needs_clarification=response.needs_clarification
)

print(f"Score: {evaluation['score']:.3f}")
```

### Batch Testing

```python
from test_bot_metrics import BotTester

# Initialize tester
tester = BotTester()

# Test multiple queries
queries = [
    {"query": "Какие характеристики у бетона М300?", "type": "informational"},
    {"query": "Хочу заказать бетон М300", "type": "order_specification"}
]

results = tester.run_batch_test(queries)
print(f"Average Score: {results['aggregate_metrics']['average_score']:.3f}")
```

### Simulated Buyer Conversation

```python
from test_bot_metrics import BotTester

# Initialize tester
tester = BotTester()

# Test with simulated buyer
result = tester.test_with_simulated_buyer(
    initial_query="Хочу заказать бетон М300",
    max_turns=5
)

print(f"Average Score: {result['average_score']:.3f}")
print(f"Turns: {result['turns']}")
```

## Output Format

### Single Query Result

```json
{
  "query": "Какие характеристики у бетона М300?",
  "bot_response": {
    "message": "...",
    "query_type": "informational",
    "needs_clarification": false
  },
  "evaluation": {
    "score": 0.85,
    "relevance": 0.28,
    "completeness": 0.27,
    "accuracy": 0.18,
    "clarity": 0.19,
    "reasoning": "..."
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Test Results

```json
{
  "results": [...],
  "aggregate_metrics": {
    "total_queries": 10,
    "average_score": 0.75,
    "min_score": 0.45,
    "max_score": 0.95,
    "scores_above_0.7": 7,
    "scores_above_0.5": 9,
    "scores_below_0.5": 1
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Metrics Explained

- **Score (0.0-1.0)**: Overall quality of the response
  - 0.9-1.0: Excellent response
  - 0.7-0.9: Good response
  - 0.5-0.7: Acceptable response
  - 0.0-0.5: Poor response

- **Relevance**: How well the response addresses the query
- **Completeness**: Whether all important information is included
- **Accuracy**: Factual correctness of the information
- **Clarity**: How clear and well-structured the response is

## Configuration

### Using Different Models

You can specify different Mistral models for evaluation and buyer simulation:

```python
tester = BotTester(
    evaluator_model="mistral-large-latest",  # For evaluation
    buyer_model="mistral-large-latest"        # For simulated buyer
)
```

Available models:
- `mistral-tiny` (fastest, cheapest)
- `mistral-small`
- `mistral-medium`
- `mistral-large-latest` (best quality, recommended for evaluation)

### Custom Buyer Persona

```python
from src.testing.simulated_buyer import SimulatedBuyer

buyer = SimulatedBuyer(
    buyer_persona="an experienced construction professional looking for high-quality materials"
)
```

## Examples

### Example 1: Quick Evaluation

```bash
python test_bot_metrics.py --queries "Нужен бетон для фундамента" "Хочу заказать 5 кубов бетона М300"
```

### Example 2: Comprehensive Testing

```bash
python test_bot_metrics.py --generate-queries 20 --simulated-buyer --save-results comprehensive_test.json
```

### Example 3: In-Memory Quick Test

```bash
python test_bot_metrics.py --generate-queries 5 --use-in-memory
```

## Requirements

- All dependencies from `requirements.txt`
- Mistral API key set in environment variable `MISTRAL_API_KEY`
- For best results, use `mistral-large-latest` for evaluation (more expensive but more accurate)

## Tips

1. **Start Small**: Begin with a few queries to test the setup
2. **Use In-Memory for Testing**: Faster iteration during development
3. **Save Results**: Always save results for analysis and comparison
4. **Monitor Costs**: Using larger models costs more API credits
5. **Review Low Scores**: Check responses with scores < 0.5 to identify issues

## Troubleshooting

### API Key Issues
- Ensure `MISTRAL_API_KEY` is set in your environment
- Check that your API key has sufficient credits

### Model Availability
- Some models may not be available in all regions
- Fall back to `mistral-small` if `mistral-large-latest` is unavailable

### Memory Issues
- Use `--use-in-memory` flag for faster testing
- Reduce number of queries if running out of memory

