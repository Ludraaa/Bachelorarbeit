import os

from src.linkers.base import BaseEntityLinker, LinkingInput, LinkingOutput
from src.linkers.entity.aqqu import surface_index_memory

DATA_DIR = os.environ.get("DATA_DIR", "data/")


class Linker(BaseEntityLinker):
    """
    Last step of the cascade: 
    FACC1 surface-form index lookup
    """

    def __init__(
        self,
        entity_list_file: str = os.path.join(
            DATA_DIR, "common_data", "facc1",
            "entity_list_file_freebase_complete_all_mention",
        ),
        surface_map_file: str = os.path.join(
            DATA_DIR, "common_data", "facc1",
            "surface_map_file_freebase_complete_all_mention",
        ),
        entity_index_prefix: str = os.path.join(
            DATA_DIR, "common_data", "facc1",
            "freebase_complete_all_mention",
        ),
        top_k: int = 50,
        facc_threshold: float = 0.001,
    ):
        self.top_k = top_k
        self.facc_threshold = facc_threshold
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory(
            entity_list_file, surface_map_file, entity_index_prefix,
        )
    
    def get_params(self) -> dict:
        return {"top_k": self.top_k, "facc_threshold": self.facc_threshold}

    def link(self, inp: LinkingInput) -> LinkingOutput:
        label_map = {}
        candidates = {}
        failed = []

        for label in inp.labels:
            mention = label.replace("_", " ")

            raw = self.surface_index.get_indexrange_entity_el_pro_one_mention(
                mention, top_k=self.top_k,
            )

            if not raw:
                failed.append(label)
                candidates[label] = []
                continue

            mids = list(raw.keys())
            scores = list(raw.values())
            cands = [(mids[0], scores[0])]
            for mid, score in zip(mids[1:], scores[1:]):
                if score >= self.facc_threshold:
                    cands.append((mid, score))

            label_map[label] = cands[0][0]
            candidates[label] = cands

        return LinkingOutput(
            label_map=label_map,
            candidates=candidates,
            failed=failed,
            debug={},
        )