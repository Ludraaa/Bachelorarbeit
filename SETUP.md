## File and Container Structure

As defined in the Dockerfile, external data can be mounted into the container. If not mounted, the produced data will not persist different docker sessions. The 3 potential mount points are:

`/workspace/data/`: all produced files (except for models and adapters) will be saved here.
`/workspace/LLMs/Models/`: base models used for finetuning go here
`/workspace/LLMs/MyModels/`: Finetuned adapters will be output here


---

## Getting base models for finetuning

To download a desired base model for finetuning, you may use HuggingFace Hub as described here.

If the desired model is gated behind license acception (like Llama-2):
`hf auth login` (this requires a token from https://huggingface.co/settings/tokens)

Example download commands of the 3 base models used:
`hf download Qwen/Qwen2.5-7B --local-dir ./LLMs/Models/Qwen2.5-7b`
`hf download meta-llama/Llama-2-7b-hf --local-dir ./LLMs/Models/Llama-2-7b`
`hf download meta-llama/Llama-2-13b-hf --local-dir ./LLMs/Models/Llama-2-13b`

---


## Freebase Setup

