# Bachelor's Thesis: Porting ChatKBQA to Wikidata

## A detailed explanation on what this project accomplishes can be found in the corresponding [paper]().

### **Original ChatKBQA Project**
- [Original ChatKBQA Repository](https://github.com/LHRLAB/ChatKBQA/tree/main)

## Usage

In order to make use of this work yourself, first clone the repository:

```bash
git clone https://github.com/Ludraaa/Bachelorarbeit.git
cd Bachelorarbeit
```

For ease of use, a docker image is provided. The commands to both build and run this can be found at the bottom of the [Dockerfile](Dockerfile). You should first take a look at [SETUP](SETUP.md). All information regarding the setup required to use this project can be found in there.

### Run Configs

In order to simplify the arguments across all steps of the pipeline, run configurations are used. Each [Makefile](Makefile) target uses a single run config. You can take a look at existing configurations in `Data/Configs/runs`.
If you want to create a new run config from scratch, all possible options are documented in the [Run config schema](Data/Configs/runs/schema.md).


## AI-Assisted Development

AI tools were used for selected non-central development tasks.
For details, see the [AI Usage Statement](AI_USAGE.md).