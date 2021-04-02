import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch


class BertSumAbsModelMigrator:
    DECODER_MAPPING: Dict[str, str] = {
        "decoder.decoder.embeddings.make_embedding.emb_luts.0.weight": "decoder.embeddings.0.weight",
        "decoder.decoder.embeddings.make_embedding.pe.pe": "decoder.embeddings.1.pe",
        "decoder.decoder.layer_norm.weight": "decoder.decoder.norm.weight",
        "decoder.decoder.layer_norm.bias": "decoder.decoder.norm.bias",
    }
    GENERATOR_MAPPING: Dict[str, str] = {
        "decoder.generator.0.weight": "decoder.generator.weight",
        "decoder.generator.0.bias": "decoder.generator.bias",
    }
    TRANSFORMER_DECODER_MAPPING: Dict[str, str] = {
        "self_attn.final_linear.weight": "self_attn.out_proj.weight",
        "self_attn.final_linear.bias": "self_attn.out_proj.bias",
        "context_attn.final_linear.weight": "multihead_attn.out_proj.weight",
        "context_attn.final_linear.bias": "multihead_attn.out_proj.bias",
        "feed_forward.w_1.weight": "linear1.weight",
        "feed_forward.w_1.bias": "linear1.bias",
        "feed_forward.w_2.weight": "linear2.weight",
        "feed_forward.w_2.bias": "linear2.bias",
        "feed_forward.layer_norm.weight": "norm3.weight",
        "feed_forward.layer_norm.bias": "norm3.bias",
        "layer_norm_1.weight": "norm1.weight",
        "layer_norm_1.bias": "norm1.bias",
        "layer_norm_2.weight": "norm2.weight",
        "layer_norm_2.bias": "norm2.bias",
    }
    DECODER_STATE_ORDER: List[str] = [
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "multihead_attn.in_proj_weight",
        "multihead_attn.in_proj_bias",
        "multihead_attn.out_proj.weight",
        "multihead_attn.out_proj.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "norm3.weight",
        "norm3.bias",
    ]

    def __init__(self) -> None:
        self.new_state_dict: Dict[str, torch.Tensor] = dict()
        self.decoder_state_dict: Dict[str, torch.Tensor] = dict()
        self.attns: Dict[str, torch.Tensor] = dict()
        self.n = 0

    def migrate(
        self,
        state_dict: Dict[str, torch.Tensor],
        migrate_transformer: bool = False,
        migrate_generator: bool = True,
    ) -> Dict[str, torch.Tensor]:
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                self.new_state_dict[k] = v
                continue

            if migrate_generator and k == "decoder.loss.one_hot":
                continue

            if migrate_generator and k in self.GENERATOR_MAPPING:
                if self.n != 0 and self.attns:
                    self._update_state_dict()
                self.new_state_dict[self.GENERATOR_MAPPING[k]] = v
                continue

            if not migrate_transformer:
                self.new_state_dict[k] = v
                continue

            if k in self.DECODER_MAPPING:
                if self.n != 0 and self.attns:
                    self._update_state_dict()
                self.new_state_dict[self.DECODER_MAPPING[k]] = v
                continue

            ks = k.split(".")[3:]
            if self.n != int(ks[0]):
                self._update_state_dict()
                self.n = int(ks[0])

            k = ".".join(ks[1:])
            if k in self.TRANSFORMER_DECODER_MAPPING:
                self.decoder_state_dict[self.TRANSFORMER_DECODER_MAPPING[k]] = v
                continue
            self.attns[k] = v

        return self.new_state_dict

    def _update_state_dict(self) -> None:
        self.decoder_state_dict["self_attn.in_proj_weight"] = torch.vstack(
            (
                self.attns["self_attn.linear_query.weight"],
                self.attns["self_attn.linear_keys.weight"],
                self.attns["self_attn.linear_values.weight"],
            )
        )
        self.decoder_state_dict["self_attn.in_proj_bias"] = torch.hstack(
            (
                self.attns["self_attn.linear_query.bias"],
                self.attns["self_attn.linear_keys.bias"],
                self.attns["self_attn.linear_values.bias"],
            )
        )
        self.decoder_state_dict["multihead_attn.in_proj_weight"] = torch.vstack(
            (
                self.attns["context_attn.linear_query.weight"],
                self.attns["context_attn.linear_keys.weight"],
                self.attns["context_attn.linear_values.weight"],
            )
        )
        self.decoder_state_dict["multihead_attn.in_proj_bias"] = torch.hstack(
            (
                self.attns["context_attn.linear_query.bias"],
                self.attns["context_attn.linear_keys.bias"],
                self.attns["context_attn.linear_values.bias"],
            )
        )

        for k in self.DECODER_STATE_ORDER:
            self.new_state_dict[
                f"decoder.decoder.layers.{self.n}.{k}"
            ] = self.decoder_state_dict[k]

        self.decoder_state_dict = dict()
        self.attns = dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    with open(input_dir / "config.json") as f:
        config = json.load(f)
    config["use_onmt_transformer"] = True
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    state_dict = torch.load(input_dir / "pytorch_model.bin", map_location="cpu")
    migrator = BertSumAbsModelMigrator()
    new_state_dict = migrator.migrate(state_dict)
    torch.save(new_state_dict, output_dir / "pytorch_model.bin")


if __name__ == "__main__":
    main()
