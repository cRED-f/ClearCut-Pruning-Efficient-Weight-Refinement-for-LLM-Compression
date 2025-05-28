%%writefile  eval.py


# Import necessary modules
import sys
import warnings
import fnmatch
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Import get_loaders function from data module within the same directory
from data import get_loaders 

from collections import defaultdict
import os

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, dataset, bs, device=torch.device("cuda:0")):
    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, bs, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    
    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        
        lm_logits = model(inputs).logits
        
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=8, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits
        

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    return ppl.item()

def eval_zero_shot(model_name, task_list=["qqp","rte","mnli","mrpc","sst2","cola", "qnli", "stsb"], 
        num_fewshot=0, use_accelerate=True, add_special_tokens=False):
    """
    Evaluate zero-shot performance using lm_eval library (v0.4.8 compatible).
    
    This function is designed to work with lm-evaluation-harness version 0.4.8.
    If you have a different version, you may need to modify this function.
    
    Args:
        model_name: Name or path of the model to evaluate
        task_list: List of tasks to evaluate on
        num_fewshot: Number of examples for few-shot evaluation (0 for zero-shot)
        use_accelerate: Whether to use accelerate for model loading
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Import required modules from lm_eval
        from lm_eval import evaluator
        import importlib
        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata
            
        try:
            lm_eval_version = metadata.version('lm-eval')
            print(f"Using lm-eval version: {lm_eval_version}")
        except:
            print("Could not determine lm-eval version")
    except ImportError:
        warnings.warn("lm_eval package not found. To use this feature, install it with: " 
                     "pip install lm-eval or pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
        print("Skipping zero-shot evaluation due to missing lm_eval package.")
        return {"error": "lm_eval package not installed", "message": "Install with: pip install lm-eval"}
    
    # Fix for super_glue dataset which requires trust_remote_code=True
    if os.environ.get('TRUST_REMOTE_CODE') != 'true':
        os.environ['TRUST_REMOTE_CODE'] = 'true'
    
    # Patch datasets.load_dataset to always use trust_remote_code=True
    try:
        import datasets
        original_load_dataset = datasets.load_dataset
        
        def patched_load_dataset(*args, **kwargs):
            if 'trust_remote_code' not in kwargs:
                kwargs['trust_remote_code'] = True
            return original_load_dataset(*args, **kwargs)
        
        datasets.load_dataset = patched_load_dataset
    except ImportError:
        print("Warning: Could not patch datasets.load_dataset - super_glue dataset might fail")
    
    print(f"Evaluating on tasks: {task_list}")
    
    # Check if this is an OPT model (OPT doesn't support use_accelerate)
    is_opt_model = "opt" in model_name.lower()
    
    # Configure model arguments - format for v0.4.8
    model_args = {
        "pretrained": model_name
    }
    
    # Only add accelerate for non-OPT models
    if use_accelerate and not is_opt_model:
        model_args["use_accelerate"] = True
        model_args["device_map"] = "auto"
    
    # Set limit for large models to avoid OOM errors
    limit = None
    if "70b" in model_name.lower() or "65b" in model_name.lower():
        limit = 2000
    
    # Build evaluation arguments for lm-eval v0.4.8
    eval_args = {
        "model": "hf",  # Use "hf" for v0.4.8
        "model_args": model_args,
        "tasks": task_list,
        "num_fewshot": num_fewshot,
        "batch_size": None  # Let the library decide the batch size
    }
    
    # Only add limit if needed
    if limit is not None:
        eval_args["limit"] = limit
    
    try:
        # Convert model_args dictionary to string format (required by lm-eval 0.4.8)
        model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])
        eval_args["model_args"] = model_args_str
        
        # Evaluate using lm-eval
        results = evaluator.simple_evaluate(**eval_args)
        
        # Print results
        print("********************************")
        print("Zero-shot evaluation results")
        
        # In v0.4.8, make_table doesn't exist, so we format the results manually
        try:
            # Try with make_table if it exists (newer versions)
            print(evaluator.make_table(results))
        except AttributeError:
            # Manual formatting for v0.4.8
            if hasattr(results, "items"):  # If results is a dict
                for task, res in results.items():
                    if task != "config":
                        print(f"{task}:")
                        if isinstance(res, dict):
                            for metric, value in res.items():
                                if isinstance(value, float):
                                    print(f"  {metric}: {value:.4f}")
                                else:
                                    print(f"  {metric}: {value}")
                        else:
                            print(f"  {res}")
            
            # Try to print average if available
            if "results" in results and "average" in results["results"]:
                print(f"\nAverage: {results['results']['average']:.4f}")
        
        return results
    except TypeError as e:
        # Handle API incompatibilities
        if "unexpected keyword argument" in str(e):
            problematic_arg = str(e).split("'")[-2]
            print(f"Removing unsupported parameter: {problematic_arg}")
            
            # If the issue is in model_args (string format), we need to fix it
            if isinstance(eval_args["model_args"], str):
                # Parse the string to remove problematic parameter
                model_args_parts = eval_args["model_args"].split(",")
                model_args_parts = [part for part in model_args_parts if problematic_arg not in part]
                eval_args["model_args"] = ",".join(model_args_parts)
            else:
                # Handle direct dictionary removal
                eval_args.pop(problematic_arg, None)
            
            # Try again with modified arguments
            results = evaluator.simple_evaluate(**eval_args)
            print("********************************")
            print("Zero-shot evaluation results")
            
            # In v0.4.8, make_table doesn't exist, so we format the results manually
            try:
                # Try with make_table if it exists (newer versions)
                print(evaluator.make_table(results))
            except AttributeError:
                # Manual formatting for v0.4.8
                if hasattr(results, "items"):  # If results is a dict
                    for task, res in results.items():
                        if task != "config":
                            print(f"{task}:")
                            if isinstance(res, dict):
                                for metric, value in res.items():
                                    if isinstance(value, float):
                                        print(f"  {metric}: {value:.4f}")
                                    else:
                                        print(f"  {metric}: {value}")
                            else:
                                print(f"  {res}")
                
                # Try to print average if available
                if "results" in results and "average" in results["results"]:
                    print(f"\nAverage: {results['results']['average']:.4f}")
            
            return results
        else:
            raise
    except ValueError as e:
        # Check for trust_remote_code error
        if "trust_remote_code=True" in str(e):
            print("Detected trust_remote_code error, trying again with explicit dataset config")
            
            # Try again with explicit dataset config that includes trust_remote_code
            try:
                from lm_eval.tasks import TaskManager
                task_manager = TaskManager()
                
                # Force trust_remote_code for all tasks
                for task in task_list:
                    if hasattr(task_manager, '_tasks') and task in task_manager._tasks:
                        task_config = task_manager._tasks[task]
                        if 'dataset_kwargs' not in task_config:
                            task_config['dataset_kwargs'] = {}
                        task_config['dataset_kwargs']['trust_remote_code'] = True
                
                # Try evaluation again
                results = evaluator.simple_evaluate(**eval_args)
                print("********************************")
                print("Zero-shot evaluation results")
                
                # In v0.4.8, make_table doesn't exist, so we format the results manually
                try:
                    # Try with make_table if it exists (newer versions)
                    print(evaluator.make_table(results))
                except AttributeError:
                    # Manual formatting for v0.4.8
                    if hasattr(results, "items"):  # If results is a dict
                        for task, res in results.items():
                            if task != "config":
                                print(f"{task}:")
                                if isinstance(res, dict):
                                    for metric, value in res.items():
                                        if isinstance(value, float):
                                            print(f"  {metric}: {value:.4f}")
                                        else:
                                            print(f"  {metric}: {value}")
                                else:
                                    print(f"  {res}")
                    
                    # Try to print average if available
                    if "results" in results and "average" in results["results"]:
                        print(f"\nAverage: {results['results']['average']:.4f}")
                
                return results
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                print("Please manually set TRUST_REMOTE_CODE=true in the environment")
                return {"error": str(e), "message": "Set TRUST_REMOTE_CODE=true"}
        
        # Handle model name errors
        if "Attempted to load model" in str(e) and "but no model for this name found" in str(e):
            print(f"Model '{eval_args['model']}' not found. Available models: {str(e).split('Supported model names: ')[1]}")
            
            # Try alternative model names
            for alt_model in ["huggingface", "hf-causal", "hf-auto"]:
                print(f"Trying alternative model name: {alt_model}")
                eval_args["model"] = alt_model
                try:
                    results = evaluator.simple_evaluate(**eval_args)
                    print("********************************")
                    print("Zero-shot evaluation results")
                    
                    # In v0.4.8, make_table doesn't exist, so we format the results manually
                    try:
                        # Try with make_table if it exists (newer versions)
                        print(evaluator.make_table(results))
                    except AttributeError:
                        # Manual formatting for v0.4.8
                        if hasattr(results, "items"):  # If results is a dict
                            for task, res in results.items():
                                if task != "config":
                                    print(f"{task}:")
                                    if isinstance(res, dict):
                                        for metric, value in res.items():
                                            if isinstance(value, float):
                                                print(f"  {metric}: {value:.4f}")
                                            else:
                                                print(f"  {metric}: {value}")
                                        else:
                                            print(f"  {res}")
                    
                    return results
                except Exception as alt_e:
                    print(f"Failed with {alt_model}: {alt_e}")
            
            return {"error": f"Model type not found: {str(e)}"}
        else:
            raise
    except Exception as e:
        print(f"Error during zero-shot evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}