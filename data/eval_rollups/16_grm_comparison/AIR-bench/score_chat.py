import argparse
import json
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import openai
import shortuuid
from func_timeout import func_set_timeout

openai.api_key = "sk-xx"
openai.api_base = "https://xx"

# --- Globals & Setup ---
MAX_API_RETRY = 5
LLM_RETRY_SLEEP = 10

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Thread-safe counters ---
lock = threading.Lock()
finish_count = 0
failed_count = 0

# --- OpenAI API Wrapper ---
@func_set_timeout(120)  # 2-minute timeout per request
def get_llm_judge_result(**kwargs):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(**kwargs)
            result = response.choices[0].message.content
            return result
        except Exception as e:
            logger.error(f"API Error (attempt {i+1}/{MAX_API_RETRY}): {e}")
            time.sleep(LLM_RETRY_SLEEP)
    raise Exception("Max retries exceeded for OpenAI API call.")

# --- Worker Task ---
def process_item(item, writer, args):
    global finish_count, failed_count

    system_prompt = (
        "You are a helpful and precise assistant for checking the quality of the answer.\n"
        "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
        "[The Start of Assistant 1's Answer]\nXAssistant1X\n[The End of Assistant 1's Answer]\n"
        "[The Start of Assistant 2's Answer]\nXAssistant2X\n[The End of Assistant 2's Answer]\n[System]\n"
        "We would like to request your feedback on the performance of two AI assistants in response to the user question "
        "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
        "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
        "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
        "The two scores are separated by a space."
    )

    # Assistant 1 is Ground Truth, Assistant 2 is the model being evaluated
    content = system_prompt.replace("XAudioX", item.get("meta_info", "")) \
                           .replace("XQuestionX", item.get("question", "")) \
                           .replace("XAssistant1X", item.get("answer_gt", "")) \
                           .replace("XAssistant2X", item.get("response", ""))

    openai_args = {
        'model': args.model_name,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'messages': [{"role": "user", "content": content}]
    }

    try:
        judge_response = get_llm_judge_result(**openai_args)
        item['gen'] = judge_response
        with lock:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            writer.flush()
            finish_count += 1
    except Exception as e:
        logger.error(f"Failed to process item {item.get('uniq_id', 'N/A')}: {e}")
        with lock:
            failed_count += 1

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Get LLM judge scores for AIR-Bench Chat results.")
    parser.add_argument("--input_file", type=str, default="./Llama2/logs/48_noise/Chat_result/Chat_result_prefix.jsonl")
    parser.add_argument("--output_file", type=str, default="./Llama2/logs/48_noise/Chat_result/Chat_result_prefix_scored.jsonl")
    parser.add_argument("--model_name", type=str, default='Pro/deepseek-ai/DeepSeek-V3')
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    # Load inference results
    with open(args.input_file, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(items)} items from {args.input_file}")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor, \
         open(args.output_file, 'w', encoding='utf-8') as writer:
        
        futures = [executor.submit(process_item, item, writer, args) for item in items]

        # Progress reporting loop
        while True:
            done_count = sum(1 for f in futures if f.done())
            logger.info(f"Progress: {done_count}/{len(futures)} | Succeeded: {finish_count} | Failed: {failed_count}")
            if done_count == len(futures):
                break
            time.sleep(10)

    logger.info(f"Scoring complete. Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
