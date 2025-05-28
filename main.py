
import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from importlib.metadata import version
from eval import eval_ppl ,eval_zero_shot
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor


def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = seqlen
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Llama model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--calib_dataset', type=str, default="c4", help='Calibration dataset')
    parser.add_argument('--eval_dataset', type=str, default="wikitext2", help='Evaluation dataset')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument("--lambda_ci", type=float, default=0.2, help="Global channel importance scaling factor")
    parser.add_argument("--sparsity_type", type=str, default="unstructured", help="Sparsity type, choose from unstructured, 4:8, 1:4, 2:4, 3:4. \
                        Please choose from the corresponding sparsity ratio")
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--prune_method", type=str, choices=[ "ria" , "ClearCut","magnitude","wanda","sparsegpt"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--semi_sparse_acc', action="store_true", help="using pytorch semi sparse acceleration. Only when sparsity type is 2:4")
    parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")
    parser.add_argument("--reconstruction", action="store_true", help="remaining weight reconstruction based on sparsegpt")
    parser.add_argument("--reallocation", action="store_true", help="Heuristic Channel Reallocation")
    parser.add_argument("--lsa", action="store_true", help="Linear Sum Assignment")
    parser.add_argument("--importance_score", type=str, default="sum", help="assign importance score for columns")
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument("--per_outneuron", action="store_true", help="pruning per outneuron. Wanda's tactic.")
    parser.add_argument("--test_bs", type=int, default=1, help="test batch size")
    parser.add_argument("--use_cusparselt", action="store_true")
    parser.add_argument("--layer_wise", action="store_true")
    parser.add_argument("--svd_threshold", type=float, default=1e-3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true", help="zero-shot performance")

    # Low-rank decomposition arguments
    parser.add_argument("--apply_low_rank", action="store_true", help="Apply low-rank decomposition after pruning")
    parser.add_argument("--rank_ratio", type=float, default=0.5, help="Ratio of singular values to keep in low-rank decomposition (0.0-1.0)")
    parser.add_argument("--target_modules", type=str, default="q,k,v,o", help="Comma-separated list of module name patterns to apply low-rank decomposition to")

    args = parser.parse_args()
    if args.use_cusparselt:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, args.seqlen)
    model.eval()
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0")
    print("use device ", device)
    print(model)
    
    if args.sparsity_ratio != 0:
        print("pruning starts")
        from prune import  prune_ria, check_sparsity , prune_ClearCut,prune_magnitude,prune_sparsegpt

        if args.prune_method == "wanda":
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "ria":
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "ClearCut":
            prune_ClearCut(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(args, model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################
    ppl_test = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.apply_low_rank:
        print("\nApplying low-rank decomposition after pruning...")
        from prune import apply_low_rank_decomposition
        
        # Parse target modules from comma-separated string
        target_modules = args.target_modules.split(",")
        
        # Apply low-rank decomposition
        model = apply_low_rank_decomposition(
            model, 
            target_modules=target_modules, 
            rank_ratio=args.rank_ratio,
        )
        
        # Evaluate perplexity after low-rank decomposition
        print("\nEvaluating model after low-rank decomposition...")
        ppl_test_lr = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
        print(f"Perplexity after low-rank decomposition: {ppl_test_lr}")
        
        # Add low-rank results to log if saving
        if args.save:
            dirname = "results/{}".format(args.model)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            filename = f"log_{args.prune_method}_lowrank.txt"
            save_filepath = os.path.join(dirname, filename)
            with open(save_filepath, "a") as f:
                print("method\tsparsity\trank_ratio\tmlp_applied\tmlp_rank_ratio\toriginal_ppl\tlowrank_ppl", file=f, flush=True)
                mlp_rank = args.mlp_rank_ratio if args.mlp_rank_ratio is not None else args.rank_ratio
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.rank_ratio:.4f}\t{args.apply_to_mlp}\t{mlp_rank:.4f}\t{ppl_test:.4f}\t{ppl_test_lr:.4f}", file=f, flush=True)

    if args.save:
        dirname = "results/{}".format(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        if args.layer_wise:
            filename = f"log_{args.prune_method}_layer.txt"
        else:
            filename = f"log_{args.prune_method}.txt"
        save_filepath = os.path.join(dirname, filename)
        with open(save_filepath, "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\treallocation\timportance_score\tlsa\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.reallocation}\t{args.importance_score}\t{args.lsa}\t{ppl_test:.4f}", file=f, flush=True)
            
            
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
    

        
    if args.eval_zero_shot:
        accelerate=True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "mnli"]
        num_shot = 0
        
        
        if args.save_model:
            results = eval_zero_shot(args.save_model, task_list, num_shot, accelerate)
        else:
            results = eval_zero_shot(args.model, task_list, num_shot, accelerate)
    

if __name__ == '__main__':
    main()