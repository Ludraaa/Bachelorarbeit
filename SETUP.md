Note: If you are running the container with the existing NFS data (accessible only on uni-freiburg PCs), all of these steps are already done. Once in the container, start Freebase using:
`python3 Freebase-Setup/virtuoso.py start 3001 -d Freebase-Setup/virtuoso_db`


## File and Container Structure

As defined in the Dockerfile, it is strongly advised to mount an external volume to the container to ensure data persistence across different sessions.
Upon mounting an empty external volume to the container, the following folder structure will appear inside:
`/extern/data/`:
    - `Configs/` - all run, training, dataset and inference configs go into the corresponding subfolders of this folder. Inside the subfolder, the configs may be organized as desired. Add new configs here to control the wanted behavior of the pipeline.
    - `Data/` - this is where the actual output files of the pipeline steps go. This folder should not have to be manually touched a lot.
    - `Freebase-Setup/` - more on this in the section below.
    - `LLMdata/` - this is where training datasets to be used by Llamafactory live. This should also not require any manual effort.
    - `Models/` - this folder stores base models to be used for finetuning. You may follow the steps in the corresponding section to download any model wanted for finetuning. Make sure the name in the training config matches the folder name of the model.
    - `MyModels/` - here live the finetuned adapters
    - `Results/` - contains the single result json file that accumulates the scores of all runs

---

## Knowledge Base Setup

While any knowledge base works in theory (even WDQS, for example), it is pretty much mandatory to use a local or at least private instance. There should be no rate limit and no super short forced query timeout. This is because the resolve step of the pipeline relies on
(in extreme cases) executing up to thousands of SPARQL queries for every single item. You will hit the rate limit on a public endpoint very quickly.

### Freebase

While any local variant works, to stay as comparable to the original ChatKBQA as possible, we use the same virtuoso setup they used. More information on that setup can be found here: https://github.com/dki-lab/Freebase-Setup/blob/master/README.md

Additionally, the FACC1 index has to be downloaded. Below are the steps as described in ChatKBQA:

- Download the mention information (including processed [FACC1](https://github.com/HXX97/GMT-KBQA/blob/main/data/common_data/facc1/README.md) mentions and all entity alias in Freebase) from [OneDrive](https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye) or [Baidu Netdisk](https://pan.baidu.com/s/1qbKP2DV1lo9jlYoBxpyTHA?pwd=qzb7) to `data/common_data/facc1/`.

```
/extern/data/ (external mount)
└── Data/
    ├── common_data/                  
        ├── facc1/   
            ├── entity_list_file_freebase_complete_all_mention
            └── surface_map_file_freebase_complete_all_mention                                           
```


### Wikidata

For Wikidata, we recommend setting up a local Qlever instance. More information can be found here: https://docs.qlever.dev/quickstart/

---

## Getting base models for finetuning

To download a desired base model for finetuning, you may use HuggingFace Hub as described here.

If the desired model is gated behind license acception (like Llama-2):
`hf auth login` (this requires a token from https://huggingface.co/settings/tokens)

Example download commands of the 3 base models used:
`hf download Qwen/Qwen2.5-7B --local-dir ./LLMs/Models/Qwen2.5-7b`
`hf download meta-llama/Llama-2-7b-hf --local-dir ./LLMs/Models/Llama-2-7b`
`hf download meta-llama/Llama-2-13b-hf --local-dir ./LLMs/Models/Llama-2-13b`