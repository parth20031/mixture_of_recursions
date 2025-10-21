import torch
import os
import gc
from pathlib import Path
import json
import argparse  # <-- Import argparse
from tqdm import tqdm

# --- LM-EVALUATION-HARNESS IMPORTS ---
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict
    from lm_eval.main import simple_evaluate
except ImportError:
    print("="*50)
    print("ERROR: lm-evaluation-harness is not installed.")
    print("Please run: pip install lm-evaluation-harness")
    print("="*50)
    exit(1)


# --- IMPORTS FROM YOUR PROJECT FILE (static_ptq.py) ---
try:
    # This assumes 'run_harness_eval.py' is in the same directory as 'static_ptq.py'
    from static_ptq import load_model_and_tokenizer, ConfigNamespace, MOR_MODEL_CLS
    print("Successfully imported functions from static_ptq.py")
except ImportError:
    print("="*50)
    print("ERROR: Could not import from 'static_ptq.py'.")
    print("Please make sure this script is in the same directory as 'static_ptq.py'")
    print("and that all its dependencies (like pyyaml) are installed.")
    print("="*50)
    exit(1)


def load_original_model_from_file(path, root_dir):
    """
    Loads the original unquantized model by calling your custom function.
    """
    print(f"--- Loading original model from: {path} ---")
    
    # This directly calls the function you wrote in static_ptq.py
    model, tokenizer = load_model_and_tokenizer(
        path, 
        'cpu',  # Harness will run on CPU
        root_dir
    )
    return model, tokenizer

def load_quantized_model_from_pt(quantized_pt_path):
    """
    Loads the complete quantized model from its .pt file.
    This is simple because the structure is saved in the file.
    """
    print(f"--- Loading quantized .pt model from: {quantized_pt_path} ---")
    
    if not os.path.exists(quantized_pt_path):
        raise FileNotFoundError(f"Quantized model file not found at: {quantized_pt_path}")

    # This one line loads the complete structure and weights
    model_quantized = torch.load(quantized_pt_path, map_location="cpu")
    model_quantized.eval() # Set to evaluation mode
    
    print("Successfully loaded full quantized model.")
    return model_quantized


def evaluate_on_harness(model, tokenizer, tasks_list):
    """
    Wraps an in-memory model and tokenizer for the lm-evaluation-harness
    and runs the specified tasks.
    """
    print(f"\n--- Starting Harness Evaluation for tasks: {tasks_list} ---")
    
    # 1. Wrap the in-memory model and tokenizer
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cpu" # Force evaluation on CPU
    )

    # 2. Get the task dictionary
    task_dict = get_task_dict(tasks_list)

    # 3. Run the evaluation
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=task_dict,
        num_fewshot=0, 
        limit=None, 
        bootstrap_iters=2,
    )
    
    print("\n--- Harness Evaluation Complete ---")
    print("Full results:")
    print(json.dumps(results['results'], indent=2))
    return results


def main_harness_compare():
    """
    Main function to orchestrate the loading, evaluation, and comparison.
    """
    # --- 1. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Compare original and quantized model accuracy.")
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to the directory of the ORIGINAL, unquantized model."
    )
    args = parser.parse_args()

    # --- 2. DEFINE PATHS AND TASKS ---
    ORIGINAL_PATH = args.checkpoint_path
    
    # --- Derive the quantized path based on our saving logic ---
    # (e.g., "path/to/model" becomes "path/to/model_quantized/quantized_model.pt")
    # Correcting your typo: "qumatized" -> "quantized"
    QUANTIZED_PATH = f"{ORIGINAL_PATH}_quantized/quantized_model.pt"
    
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Choose which benchmarks to run. Start with 1-3. MMLU is very slow.
    TASKS_TO_RUN = [
        'hellaswag', 
        'arc_easy', 
        'winogrande'
    ]
    # You can add more from your list: 
    # 'piqa', 'arc_challenge', 'openbookqa', 'truthfulqa', 'mmlu', 'lambada_openai'
    
    print("="*50)
    print(f"Starting comparison for tasks: {TASKS_TO_RUN}")
    print(f"Original:  {ORIGINAL_PATH}")
    print(f"Quantized: {QUANTIZED_PATH}")
    print("="*50)

    # --- 3. Evaluate Original Model ---
    original_results = None
    tokenizer = None
    try:
        original_model, tokenizer = load_original_model_from_file(ORIGINAL_PATH, PROJECT_ROOT)
        original_results = evaluate_on_harness(original_model, tokenizer, TASKS_TO_RUN)
        
        del original_model
        gc.collect()
        print("Original model unloaded from memory.")

    except Exception as e:
        print(f"\n--- CRITICAL ERROR evaluating original model: {e} ---")
        return

    # --- 4. Evaluate Quantized Model ---
    quantized_results = None
    if tokenizer is None:
        print("Error: Tokenizer was not loaded. Cannot proceed.")
        return
        
    try:
        quantized_model = load_quantized_model_from_pt(QUANTIZED_PATH)
        quantized_results = evaluate_on_harness(quantized_model, tokenizer, TASKS_TO_RUN)
        
        del quantized_model
        gc.collect()
        print("Quantized model unloaded from memory.")
        
    except Exception as e:
        print(f"\n--- CRITICAL ERROR evaluating quantized model: {e} ---")
        print(f"Ensure the file exists at: {QUANTIZED_PATH}")
        return

    # --- 5. Print Final Comparison Table ---
    if original_results is None or quantized_results is None:
        print("\nComparison failed, one model did not produce results.")
        return

    print("\n" + "="*70)
    print(" " * 20 + "HARNESS ACCURACY COMPARISON")
    print("="*7Example)
    
    print(f"{'Benchmark':<18} | {'Metric':<10} | {'Original':<10} | {'Quantized':<10} | {'Change':<10}")
    print("-" * 70)

    for task in TASKS_TO_RUN:
        try:
            # Find the main accuracy metric (e.g., 'acc_norm' or 'acc')
            metric_name = 'acc_norm'
            if metric_name not in original_results['results'][task]:
                metric_name = 'acc'
            
            orig_acc = original_results['results'][task][metric_name]
            quant_acc = quantized_results['results'][task][metric_name]
            change = quant_acc - orig_acc
            
            print(f"{task:<18} | {metric_name:<10} | {orig_acc:<10.4f} | {quant_acc:<10.4f} | {change:<+10.4f}")
            
        except KeyError:
            print(f"Could not find a comparable metric for task '{task}'. Skipping.")
    
    print("="*70)
    print("✅ Comparison complete.")


if __name__ == "__main__":
    main_harness_compare()