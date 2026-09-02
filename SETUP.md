## File Structure

As defined in the Dockerfile, large external data must be mounted into the container at:

`/extern/data/`

This prevents data loss and avoids copying large assets into the image.

Inside the container, write access is only available in:

`/workspace/output/`


By default, output is written there via:

```
export OUTPUT=/workspace/output/
```

You may override this to point to an externally mounted output volume if desired.

---

## Requirements

In addition to the datasets and models you intend to use, the following must be installed manually.

All installations must be done on the host machine, not inside the container, and then mounted into the container.

---


### ReFinEd

If you are running on Uni Freiburg’s Tagus and mounted the correct data folder, ReFinEd is already available at:

`/extern/data/Models/refined`

In this case, simply set:

```
export REFINED_PATH=/extern/data/Models/refined/
```


If you are running externally and do not have ReFinEd installed, download it manually.

Note: ReFinEd is very large. Install it on the host machine, ideally in your external data directory, and mount it into the container.

```bash
mkdir -p <your_directory>

curl https://almond-static.stanford.edu/research/qald/refined-finetune/config.json \
  -o <your_directory>/config.json

curl https://almond-static.stanford.edu/research/qald/refined-finetune/model.pt \
  -o <your_directory>/model.pt

curl https://almond-static.stanford.edu/research/qald/refined-finetune/precomputed_entity_descriptions_emb_wikidata_33831487-300.np \
  -o <your_directory>/precomputed_entity_descriptions_emb_wikidata_33831487-300.np
```
Once downloaded, mount the directory into the container and set:

`export REFINED_PATH=/extern/data/path/to/refined/`

---


### MongoDB

MongoDB is required by eval.py.

If you are running on Uni Freiburg’s Tagus and mounted the correct data folder, MongoDB is already available at:

`/extern/data/mongo)`

In this case, simply set:

```
export MONGODB_PATH=/extern/data/mongo/bin/mongod
```



MongoDB must be installed on the host machine and mounted into the container for reuse.
Do NOT install MongoDB inside the container.

#### Manual binary installation

```bash
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.5.tgz
tar -xzf mongodb-linux-x86_64-ubuntu2204-7.0.5.tgz
rm mongodb-linux-x86_64-ubuntu2204-7.0.5.tgz
```

# Optional but recommended
```
mv mongodb-linux-x86_64-ubuntu2204-7.0.5 mongo
```

Place this directory in your external data mount (e.g. `/extern/data/mongo`).

Then set the path to the mongod binary:

```
export MONGOD=/extern/data/path/to/mongo/bin/mongod
```


MongoDB will always write its database files to:

`$(OUTPUT)/Mongo/`

By default:

`/workspace/output/Mongo/`

