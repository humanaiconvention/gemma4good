import json
import random

def generate_synthetic_dpo_dataset(input_file, output_file):
    """
    Simulates the PRISM geometry rejection-sampling process.
    In production, this would run the base model to generate N completions,
    score each with PRISM outlier_geometry(), and output the lowest 'qh' as chosen,
    highest 'qh' as rejected.
    """
    print(f"Reading HAIC dataset from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Create mock data.")
        data = [{
            "system": "You are a governed AI.",
            "prompt": "Evaluate this condition.",
            "response": "The condition appears stable under governance review."
        }]
        
    dpo_dataset = []
    
    for item in data:
        prompt = item.get("prompt", "")
        system = item.get("system", "")
        
        # The 'chosen' response is the geometrically stable HAIC completion
        chosen = item.get("response", "")
        
        # The 'rejected' response simulates a completion that triggers high kurtosis/qh
        # (e.g., standard RLHF slop or non-structured output)
        rejected = chosen + " I hope this helps you out today! Let me know if you need anything else by reaching out directly to the server."
        
        dpo_row = {
            "system": system,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        dpo_dataset.append(dpo_row)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in dpo_dataset:
            f.write(json.dumps(row) + "\\n")
            
    print(f"Generated {len(dpo_dataset)} DPO pairs to {output_file}")

if __name__ == "__main__":
    generate_synthetic_dpo_dataset(
        input_file=r"d:\gemma4good\grounding_gemma4_v2.jsonl",
        output_file=r"D:\gemma4good\experiments\phase3\prism_dpo_pairs.jsonl"
    )
