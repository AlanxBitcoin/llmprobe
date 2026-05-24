from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - ProbePipeline coordinates end-to-end study/probe workflows.
# - Hidden states should come through extract_hidden/hidden_store interfaces.
# - Output artifacts are written under data/outputs.
# - Pipeline remains orchestration layer; probe training logic stays in src/probes.

import hashlib
import html
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .probes.attribute_probe import build_feature_bank, fit_full_attribute_probes, load_attribute_rows, predict_word_attributes
from .probes.concept_match import ConceptMatcher
from .utils.extract_hidden import extract_sequence_positional_states, extract_word_hidden_states, summarize_top_dims
from .probes.symbolic_attributes import SymbolicAttributeRegistry
from .utils.utils import chunked, ensure_dir, read_lines, safe_stem, write_csv, write_json, write_text
from .utils.video import preferred_video_suffix, synthesize_video
from .utils.visualize_multi_word import render_multi_word_dashboard
from .utils.visualize_single_word import render_single_word_dashboard
from .utils.visualize_color_experiment import (
    render_per_word_dim_stats,
    render_all_input_dim_stats,
    render_three_mode_comparison,
    render_full_4096_mean_landscape,
    render_full_4096_top_signed_bars,
)


class ProbePipeline:
    def __init__(self, config: dict[str, Any], bundle) -> None:
        self.config = config
        self.bundle = bundle
        self.base_output = ensure_dir(config["output"]["base_dir"])
        self.matcher = ConceptMatcher(
            bundle=bundle,
            concept_type_file="data/concept_types.txt",
            concept_value_file="data/concept_values.txt",
        )
        self.symbolic_matcher = ConceptMatcher(
            bundle=bundle,
            concept_type_file="data/symbolic_concept_types.txt",
            concept_value_file="data/symbolic_concept_values.txt",
            catalog_path="data/symbolic_concept_catalog.yaml",
        )
        self.symbolic_registry = SymbolicAttributeRegistry("data/symbolic_token_attributes.yaml")
        self._token_lists = {
            "english_punctuation": set(read_lines("data/english_punctuation.txt")),
            "chinese_punctuation": set(read_lines("data/chinese_punctuation.txt")),
            "arabic_digits": set(read_lines("data/arabic_digits.txt")),
            "roman_digits": set(read_lines("data/roman_digits.txt")),
        }
        self._attribute_probe_cache: dict[str, Any] | None = None

    def run_single_word(self, word: str) -> dict[str, Any]:
        result, _vector = self._run_single_word_with_vector(word)
        return result

    def _run_single_word_with_vector(self, word: str) -> tuple[dict[str, Any], np.ndarray]:
        analysis_cfg = self.config["analysis"]
        target_layer = int(analysis_cfg["target_layer"])
        hidden = extract_word_hidden_states(self.bundle, word, config=self.config)
        layer_vector = np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32)
        result = self._build_vector_result(
            word=word,
            vector=layer_vector,
            tokens=hidden["tokens"],
            include_attribute_predictions=True,
        )
        return result, layer_vector

    def run_color_words_experiment(self, word_file: str = "data/color_words.txt", run_name: str = "color_words") -> dict[str, Any]:
        words = read_lines(word_file)
        if not words:
            raise ValueError(f"No color words found in {word_file!r}.")

        run_dir = ensure_dir(self.base_output / run_name)
        per_word_run = f"{run_name}/single_words"
        all_input_run = f"{run_name}/all_input"
        positional_run = f"{run_name}/positional_words"
        target_layer = int(self.config["analysis"]["target_layer"])
        dpi = int(self.config["visualization"]["figure_dpi"])
        frames: list[str] = []
        per_word_rows: list[dict[str, Any]] = []
        per_word_index_rows: list[dict[str, Any]] = []
        per_word_vectors: list[np.ndarray] = []

        for index, word in enumerate(words, start=1):
            result, vector = self._run_single_word_with_vector(word)
            per_word_vectors.append(vector)
            frames.append(self.save_single_word_outputs(result, per_word_run, frame_index=index))
            per_word_rows.append(self._color_dim_summary_row(index, result))
            per_word_index_rows.append(self._single_summary_row(index, result))

        per_word_average_rows = self._color_dim_average_rows(per_word_rows)

        all_text = " ".join(words)
        all_result, all_input_vector = self._run_single_word_with_vector(all_text)
        all_result["input_text"] = all_text
        all_result["word"] = "all_color_words"
        all_frame = self.save_single_word_outputs(all_result, all_input_run, frame_index=1)
        all_rows = [self._color_dim_summary_row(1, all_result)]
        all_index_rows = [self._single_summary_row(1, all_result)]

        # ── All-input dim stats (single combined run) ──────────────────────
        all_input_top_dims = all_result.get("top_dims", [])
        all_input_dim_stats_rows = self._all_input_dim_stats_rows(all_result)

        # ── Positional mode (one forward pass, per-word extraction) ────────
        pos_results = extract_sequence_positional_states(self.bundle, words, target_layer)
        pos_frames: list[str] = []
        pos_dim_rows: list[dict[str, Any]] = []
        pos_index_rows: list[dict[str, Any]] = []
        positional_vectors: list[np.ndarray] = []
        for idx, pos_res in enumerate(pos_results, start=1):
            vector = np.asarray(pos_res["vector"], dtype=np.float32)
            positional_vectors.append(vector)
            result = self._build_positional_result(
                word=pos_res["word"],
                tokens=pos_res["tokens"],
                token_positions=pos_res["token_positions"],
                last_token_pos=pos_res["last_token_pos"],
                vector=vector,
                target_layer=target_layer,
            )
            pos_frames.append(self.save_single_word_outputs(result, positional_run, frame_index=idx))
            pos_dim_rows.append(self._color_dim_summary_row(idx, result))
            pos_index_rows.append(self._single_summary_row(idx, result))
        pos_average_rows = self._color_dim_average_rows(pos_dim_rows)

        per_word_video = self._finalize_video(per_word_run, frames)
        all_input_video = self._finalize_video(all_input_run, [all_frame])
        positional_video = self._finalize_video(positional_run, pos_frames)
        self._write_run_index(per_word_run, per_word_index_rows, mode="single", video_path=per_word_video)
        self._write_run_index(all_input_run, all_index_rows, mode="single", video_path=all_input_video)
        self._write_run_index(positional_run, pos_index_rows, mode="single", video_path=positional_video)

        # ── Dimension statistics charts ────────────────────────────────────
        per_word_stats_chart = run_dir / "per_word_dim_stats_chart.png"
        all_input_stats_chart = run_dir / "all_input_dim_stats_chart.png"
        positional_stats_chart = run_dir / "positional_dim_stats_chart.png"
        comparison_chart = run_dir / "three_mode_comparison_chart.png"

        render_per_word_dim_stats(
            average_rows=per_word_average_rows,
            output_path=per_word_stats_chart,
            top_k=20,
            dpi=dpi,
        )
        render_all_input_dim_stats(
            top_dims=all_input_top_dims,
            output_path=all_input_stats_chart,
            word="all color words",
            layer_index=target_layer,
            dpi=dpi,
        )
        render_per_word_dim_stats(
            average_rows=pos_average_rows,
            output_path=positional_stats_chart,
            top_k=20,
            dpi=dpi,
        )
        render_three_mode_comparison(
            per_word_average_rows=per_word_average_rows,
            all_input_top_dims=all_input_top_dims,
            positional_average_rows=pos_average_rows,
            output_path=comparison_chart,
            top_k=20,
            layer_index=target_layer,
            dpi=dpi,
        )

        write_csv(run_dir / "per_word_dim_extremes.csv", per_word_rows)
        write_json(run_dir / "per_word_dim_extremes.json", {"word_file": word_file, "rows": per_word_rows})
        write_csv(run_dir / "per_word_dim_average_abs.csv", per_word_average_rows)
        write_json(run_dir / "per_word_dim_average_abs.json", {"word_file": word_file, "rows": per_word_average_rows})
        write_csv(run_dir / "all_input_dim_extremes.csv", all_rows)
        write_json(run_dir / "all_input_dim_extremes.json", {"word_file": word_file, "input": all_text, "rows": all_rows})
        write_csv(run_dir / "all_input_dim_stats.csv", all_input_dim_stats_rows)
        write_json(run_dir / "all_input_dim_stats.json", {"word_file": word_file, "input": all_text, "rows": all_input_dim_stats_rows})
        write_csv(run_dir / "positional_dim_extremes.csv", pos_dim_rows)
        write_json(run_dir / "positional_dim_extremes.json", {"word_file": word_file, "rows": pos_dim_rows})
        write_csv(run_dir / "positional_dim_average_abs.csv", pos_average_rows)
        write_json(run_dir / "positional_dim_average_abs.json", {"word_file": word_file, "rows": pos_average_rows})
        per_word_full_rows = self._full_dim_average_rows(per_word_vectors)
        all_input_full_rows = self._single_vector_full_dim_rows(all_input_vector)
        positional_full_rows = self._full_dim_average_rows(positional_vectors)
        full_4096_landscape_chart = run_dir / "full_4096_mean_landscape.png"
        full_4096_top_bars_chart = run_dir / "full_4096_top_signed_bars.png"
        render_full_4096_mean_landscape(
            per_word_rows=per_word_full_rows,
            all_input_rows=all_input_full_rows,
            positional_rows=positional_full_rows,
            output_path=full_4096_landscape_chart,
            layer_index=target_layer,
            dpi=dpi,
        )
        render_full_4096_top_signed_bars(
            per_word_rows=per_word_full_rows,
            all_input_rows=all_input_full_rows,
            positional_rows=positional_full_rows,
            output_path=full_4096_top_bars_chart,
            top_k=20,
            layer_index=target_layer,
            dpi=dpi,
        )
        write_csv(run_dir / "per_word_full_4096_dim_stats.csv", per_word_full_rows)
        write_json(
            run_dir / "per_word_full_4096_dim_stats.json",
            {"word_file": word_file, "mode": "per_word", "sample_count": len(per_word_vectors), "rows": per_word_full_rows},
        )
        write_csv(run_dir / "all_input_full_4096_dim_stats.csv", all_input_full_rows)
        write_json(
            run_dir / "all_input_full_4096_dim_stats.json",
            {"word_file": word_file, "mode": "all_input_last_token", "sample_count": 1, "input": all_text, "rows": all_input_full_rows},
        )
        write_csv(run_dir / "positional_full_4096_dim_stats.csv", positional_full_rows)
        write_json(
            run_dir / "positional_full_4096_dim_stats.json",
            {"word_file": word_file, "mode": "positional", "sample_count": len(positional_vectors), "rows": positional_full_rows},
        )
        write_text(
            run_dir / "index.html",
            self._build_color_experiment_html(
                run_name=run_name,
                word_count=len(words),
                per_word_video=self._relative_run_path(str(per_word_video) if per_word_video else None, run_dir),
                all_input_video=self._relative_run_path(str(all_input_video) if all_input_video else None, run_dir),
                positional_video=self._relative_run_path(str(positional_video) if positional_video else None, run_dir),
            ),
        )
        return {
            "run_dir": str(run_dir),
            "per_word_video": str(per_word_video) if per_word_video else None,
            "all_input_video": str(all_input_video) if all_input_video else None,
            "positional_video": str(positional_video) if positional_video else None,
            "word_count": len(words),
            "per_word_stats_chart": str(per_word_stats_chart),
            "all_input_stats_chart": str(all_input_stats_chart),
            "positional_stats_chart": str(positional_stats_chart),
            "comparison_chart": str(comparison_chart),
            "full_4096_landscape_chart": str(full_4096_landscape_chart),
            "full_4096_top_bars_chart": str(full_4096_top_bars_chart),
        }

    def run_combined_word_sum(self, words: list[str]) -> dict[str, Any]:
        if len(words) < 2:
            raise ValueError("Need at least two words to build a combined sum.")
        hidden_per_word = [extract_word_hidden_states(self.bundle, word, config=self.config) for word in words]
        target_layer = int(self.config["analysis"]["target_layer"])
        vectors = [np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32) for hidden in hidden_per_word]
        combined_vector = np.sum(np.stack(vectors, axis=0), axis=0)
        result = self._build_vector_result(
            word=" + ".join(words),
            vector=combined_vector,
            tokens=words,
            include_attribute_predictions=False,
        )
        result["source_words"] = words
        result["operation"] = "sum"
        return result

    def run_combined_word_diff(self, minuend: str, subtrahend: str) -> dict[str, Any]:
        target_layer = int(self.config["analysis"]["target_layer"])
        left = extract_word_hidden_states(self.bundle, minuend, config=self.config)
        right = extract_word_hidden_states(self.bundle, subtrahend, config=self.config)
        left_vector = np.asarray(left["layers"][target_layer]["vector"], dtype=np.float32)
        right_vector = np.asarray(right["layers"][target_layer]["vector"], dtype=np.float32)
        combined_vector = left_vector - right_vector
        result = self._build_vector_result(
            word=f"{minuend} - {subtrahend}",
            vector=combined_vector,
            tokens=[minuend, subtrahend],
            include_attribute_predictions=False,
        )
        result["source_words"] = [minuend, subtrahend]
        result["operation"] = "diff"
        return result

    def _build_vector_result(
        self,
        word: str,
        vector: np.ndarray,
        tokens: list[str],
        include_attribute_predictions: bool,
    ) -> dict[str, Any]:
        analysis_cfg = self.config["analysis"]
        target_layer = int(analysis_cfg["target_layer"])
        top_dims = summarize_top_dims(vector, int(analysis_cfg["top_k_dims"]))
        token_profile = self._classify_token(word)
        matcher = self._matcher_for_profile(token_profile)
        if token_profile["attribute_probe_enabled"]:
            matches = matcher.match(vector, int(analysis_cfg["top_k_concepts"])) if matcher else {"types": [], "values": [], "catalog_types": []}
            dimension_semantics = matcher.explain_top_dims(vector, top_dims, top_k=3) if matcher else []
        else:
            matches = {"types": [], "values": [], "catalog_types": []}
            dimension_semantics = []
        attribute_predictions = None
        if include_attribute_predictions and token_profile["attribute_probe_enabled"]:
            attribute_predictions = self._predict_attributes(word, target_layer, token_profile)
        return {
            "word": word,
            "target_layer": target_layer,
            "tokens": tokens,
            "top_dims": top_dims,
            "top_max_dims": self._top_max_dims_abs(vector, top_k=10),
            "top_min_dims": self._top_min_dims_abs(vector, top_k=10),
            "matches": matches,
            "dimension_semantics": dimension_semantics,
            "attribute_predictions": attribute_predictions,
            "token_profile": token_profile,
        }

    def save_single_word_outputs(self, result: dict[str, Any], run_name: str, frame_index: int | None = None) -> str:
        figures_dir = ensure_dir(self.base_output / run_name / "figures")
        data_dir = ensure_dir(self.base_output / run_name / "data")
        stem_core = safe_stem(result["word"])
        stem = f"{frame_index:04d}_{stem_core}" if frame_index is not None else stem_core
        figure_path = figures_dir / f"{stem}.png"
        json_path = data_dir / f"{stem}.json"
        render_single_word_dashboard(
            word=result["word"],
            layer_index=result["target_layer"],
            top_dims=result["top_dims"],
            matches=result["matches"],
            attribute_predictions=result.get("attribute_predictions"),
            dimension_semantics=result.get("dimension_semantics"),
            token_profile=result.get("token_profile"),
            output_path=figure_path,
            dpi=int(self.config["visualization"]["figure_dpi"]),
            cmap=self.config["visualization"]["cmap"],
        )
        if self.config["output"].get("save_json", True):
            write_json(json_path, result)
        return str(figure_path)

    def run_multi_word(self, words: list[str]) -> dict[str, Any]:
        hidden_per_word = [extract_word_hidden_states(self.bundle, word, config=self.config) for word in words]
        layer_count = len(hidden_per_word[0]["layers"])
        layer_strengths = np.zeros((layer_count, len(words)), dtype=np.float32)
        for word_idx, hidden in enumerate(hidden_per_word):
            for layer_info in hidden["layers"]:
                layer_idx = int(layer_info["layer"])
                vector = np.asarray(layer_info["vector"])
                layer_strengths[layer_idx, word_idx] = float(np.mean(np.abs(vector)))
        salient_paths = self._build_salient_paths(words, layer_strengths)
        return {
            "words": words,
            "layer_strengths": layer_strengths.tolist(),
            "salient_paths": salient_paths,
        }

    def _build_salient_paths(self, words: list[str], layer_strengths: np.ndarray) -> list[tuple[str, str, float]]:
        paths: list[tuple[str, str, float]] = []
        for layer_idx in range(layer_strengths.shape[0] - 1):
            current = layer_strengths[layer_idx]
            nxt = layer_strengths[layer_idx + 1]
            if current.max() == 0 or nxt.max() == 0:
                continue
            current_norm = current / current.max()
            next_norm = nxt / nxt.max()
            candidates: list[tuple[str, str, float]] = []
            for src_idx, src_word in enumerate(words):
                for dst_idx, dst_word in enumerate(words):
                    continuity_bonus = 0.12 if src_idx == dst_idx else 0.0
                    proximity_bonus = 0.06 if abs(src_idx - dst_idx) == 1 else 0.0
                    weight = float(current_norm[src_idx] * next_norm[dst_idx] + continuity_bonus + proximity_bonus)
                    candidates.append((f"L{layer_idx}:{src_word}", f"L{layer_idx + 1}:{dst_word}", weight))
            candidates.sort(key=lambda item: item[2], reverse=True)
            paths.extend(candidates[: max(2, len(words))])
        return paths

    def save_multi_word_outputs(self, result: dict[str, Any], run_name: str, frame_index: int | None = None) -> str:
        figures_dir = ensure_dir(self.base_output / run_name / "figures")
        data_dir = ensure_dir(self.base_output / run_name / "data")
        stem_core = safe_stem("_".join(result["words"]))
        stem = f"{frame_index:04d}_{stem_core}" if frame_index is not None else stem_core
        figure_path = figures_dir / f"{stem}.png"
        json_path = data_dir / f"{stem}.json"
        render_multi_word_dashboard(
            words=result["words"],
            layer_strengths=np.asarray(result["layer_strengths"]),
            salient_paths=[tuple(item) for item in result["salient_paths"]],
            output_path=figure_path,
            dpi=int(self.config["visualization"]["figure_dpi"]),
            cmap=self.config["visualization"]["cmap"],
        )
        if self.config["output"].get("save_json", True):
            write_json(json_path, result)
        return str(figure_path)

    def run_single_batch(self, word_file: str | None = None, run_name: str = "single_batch") -> Path | None:
        words = read_lines(word_file or self.config["input"]["word_file"])
        frames: list[str] = []
        summary_rows: list[dict[str, Any]] = []
        for index, word in enumerate(words, start=1):
            result = self.run_single_word(word)
            frames.append(self.save_single_word_outputs(result, run_name, frame_index=index))
            summary_rows.append(self._single_summary_row(index, result))
        video_path = self._finalize_video(run_name, frames)
        self._write_run_index(run_name, summary_rows, mode="single", video_path=video_path)
        return video_path

    def run_multi_batch(self, batch_size: int, word_file: str | None = None, run_name: str | None = None) -> Path | None:
        words = read_lines(word_file or self.config["input"]["word_file"])
        run_name = run_name or f"multi_batch_{batch_size}"
        frames: list[str] = []
        summary_rows: list[dict[str, Any]] = []
        for index, batch in enumerate(chunked(words, batch_size), start=1):
            if len(batch) < 2:
                continue
            result = self.run_multi_word(batch)
            frames.append(self.save_multi_word_outputs(result, run_name, frame_index=index))
            summary_rows.append(self._multi_summary_row(index, result))
        video_path = self._finalize_video(run_name, frames)
        self._write_run_index(run_name, summary_rows, mode="multi", video_path=video_path)
        return video_path

    def run_dimension_report(self, dims: list[int] | None = None, word_file: str | None = None) -> dict[str, Any]:
        target_layer = int(self.config["analysis"]["target_layer"])
        top_k_words = int(self.config["analysis"].get("top_k_words_per_dim", 8))
        top_k_concepts = int(self.config["analysis"].get("top_k_concepts_per_axis", 5))
        
        words = read_lines(word_file or self.config["input"]["word_file"])
        attribute_rows = load_attribute_rows("data/word_attributes.csv")
        attribute_map = {row["word"]: row for row in attribute_rows}
        for token, attrs in self.symbolic_registry.token_map.items():
            attribute_map[token] = attrs
        matcher = self._matcher_for_words(words)

        vectors: dict[str, np.ndarray] = {}
        for word in words:
            hidden = extract_word_hidden_states(self.bundle, word, config=self.config)
            vectors[word] = np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32)

        if not dims:
            dims = self._auto_select_salient_dims(vectors, int(self.config["analysis"]["top_k_dims"]))

        reports: list[dict[str, Any]] = []
        for dim in dims:
            scored = [{"word": word, "value": float(vector[dim])} for word, vector in vectors.items()]
            scored.sort(key=lambda item: item["value"], reverse=True)
            positive_axis_concepts, positive_axis_types = matcher.explain_dimension_axis(dim, direction="positive", top_k=top_k_concepts)
            negative_axis_concepts, negative_axis_types = matcher.explain_dimension_axis(dim, direction="negative", top_k=top_k_concepts)
            reports.append(
                {
                    "dim": dim,
                    "top_positive_words": scored[:top_k_words],
                    "top_negative_words": list(reversed(scored[-top_k_words:])),
                    "positive_axis": {"types": positive_axis_types, "concepts": positive_axis_concepts},
                    "negative_axis": {"types": negative_axis_types, "concepts": negative_axis_concepts},
                    "attribute_group_means": self._dimension_attribute_group_means(dim, scored, attribute_map),
                }
            )

        return {
            "target_layer": target_layer,
            "word_count": len(words),
            "word_file": word_file or self.config["input"]["word_file"],
            "selected_dims": dims,
            "dims": reports,
        }

    def run_global_analysis(self, word_file: str, run_name: str, batch_size: int = 2) -> dict[str, str | None]:
        single_video = self.run_single_batch(word_file=word_file, run_name=f"{run_name}_single")
        multi_video = self.run_multi_batch(batch_size=batch_size, word_file=word_file, run_name=f"{run_name}_multi_{batch_size}")
        dim_report = self.run_dimension_report(word_file=word_file)
        dim_report_path = self.save_dimension_report(dim_report, run_name=f"{run_name}_dimension_report")
        return {
            "single_video": str(single_video) if single_video else None,
            "multi_video": str(multi_video) if multi_video else None,
            "dimension_report": str(dim_report_path),
        }

    def run_global_analysis_all(
        self,
        word_files: list[str] | None = None,
        run_name: str = "global_analysis_all",
        batch_size: int = 2,
        word_dir: str = "data",
        pattern: str = "*.txt",
    ) -> Path:
        discovered = self._resolve_word_files(word_files=word_files, word_dir=word_dir, pattern=pattern)
        if not discovered:
            raise ValueError(f"No word list files found from word_dir={word_dir!r} pattern={pattern!r}")

        run_dir = ensure_dir(self.base_output / run_name)
        rows: list[dict[str, Any]] = []
        for index, word_path in enumerate(discovered, start=1):
            words = read_lines(word_path)
            file_stem = safe_stem(word_path.stem)
            relative_word_file = word_path.as_posix()
            profile = self._build_word_list_profile(words)
            row: dict[str, Any] = {
                "index": index,
                "list_name": file_stem,
                "word_file": relative_word_file,
                "word_count": len(words),
                "attribute_probe_words": profile["attribute_probe_words"],
                "list_profile": profile["list_profile"],
                "token_classes": profile["token_classes"],
                "sample_words": ", ".join(words[:5]),
                "status": "ok",
                "error": "",
                "single_index": "",
                "single_overview": "",
                "single_video": "",
                "multi_index": "",
                "multi_overview": "",
                "multi_video": "",
                "dimension_report": "",
            }
            try:
                outputs = self.run_global_analysis(str(word_path), f"{run_name}/{file_stem}", batch_size=batch_size)
                row["single_index"] = f"{file_stem}_single/index.html"
                row["single_overview"] = f"{file_stem}_single/overview.html"
                row["multi_index"] = f"{file_stem}_multi_{batch_size}/index.html"
                row["multi_overview"] = f"{file_stem}_multi_{batch_size}/overview.html"
                row["single_video"] = self._relative_run_path(outputs.get("single_video"), run_dir)
                row["multi_video"] = self._relative_run_path(outputs.get("multi_video"), run_dir)
                row["dimension_report"] = self._relative_run_path(outputs.get("dimension_report"), run_dir)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            rows.append(row)

        self._write_global_analysis_bundle(run_name, rows, batch_size=batch_size, word_dir=word_dir, pattern=pattern)
        return run_dir / "index.html"

    def run_word_family_dim_report(self, words: list[str]) -> dict[str, Any]:
        if len(words) < 2:
            raise ValueError("Need at least two words for a family dimension report.")
        results = [self.run_single_word(word) for word in words]
        results.append(self.run_combined_word_sum(words))
        if len(words) == 2:
            results.append(self.run_combined_word_diff(words[0], words[1]))
        dims: list[int] = []
        seen: set[int] = set()
        for result in results:
            for item in result["top_dims"]:
                dim = int(item["dim"])
                if dim not in seen:
                    seen.add(dim)
                    dims.append(dim)
        report = self.run_dimension_report(dims)
        report["source_words"] = words
        report["source_runs"] = [result["word"] for result in results]
        return report

    def run_word_contrast_report(self, left_word: str, right_word: str) -> dict[str, Any]:
        target_layer = int(self.config["analysis"]["target_layer"])
        left_hidden = extract_word_hidden_states(self.bundle, left_word, config=self.config)
        right_hidden = extract_word_hidden_states(self.bundle, right_word, config=self.config)
        left_vector = np.asarray(left_hidden["layers"][target_layer]["vector"], dtype=np.float32)
        right_vector = np.asarray(right_hidden["layers"][target_layer]["vector"], dtype=np.float32)
        diff_vector = left_vector - right_vector

        shared_score = np.minimum(np.abs(left_vector), np.abs(right_vector))
        diff_score = np.abs(diff_vector)
        top_k = int(self.config["analysis"]["top_k_dims"])
        shared_dims = np.argsort(shared_score)[::-1][:top_k]
        diff_dims = np.argsort(diff_score)[::-1][:top_k]

        matcher = self._matcher_for_words([left_word, right_word])
        shared_rows = [self._contrast_dim_row(int(dim), left_vector, right_vector, diff_vector, mode="shared", matcher=matcher) for dim in shared_dims]
        diff_rows = [self._contrast_dim_row(int(dim), left_vector, right_vector, diff_vector, mode="diff", matcher=matcher) for dim in diff_dims]
        return {
            "target_layer": target_layer,
            "left_word": left_word,
            "right_word": right_word,
            "shared_dims": shared_rows,
            "diff_dims": diff_rows,
        }

    def save_word_contrast_report(self, report: dict[str, Any], run_name: str = "word_contrast_report") -> Path:
        run_dir = ensure_dir(self.base_output / run_name)
        stem = safe_stem(f"{report['left_word']}_{report['right_word']}")
        json_path = run_dir / f"{stem}.json"
        html_path = run_dir / f"{stem}.html"
        write_json(json_path, report)
        write_text(html_path, self._build_word_contrast_html(report))
        return html_path

    def save_dimension_report(self, report: dict[str, Any], run_name: str = "dimension_report") -> Path:
        run_dir = ensure_dir(self.base_output / run_name)
        dims_key = ",".join(str(item["dim"]) for item in report["dims"])
        digest = hashlib.sha1(dims_key.encode("utf-8")).hexdigest()[:12]
        if report.get("source_words"):
            source_key = safe_stem("_".join(report["source_words"]))
            stem = f"{source_key}_{digest}"
        else:
            stem = f"dims_{digest}"
        json_path = run_dir / f"{stem}.json"
        html_path = run_dir / f"{stem}.html"
        write_json(json_path, report)
        write_text(html_path, self._build_dimension_report_html(report))
        return html_path

    def _finalize_video(self, run_name: str, frames: list[str]) -> Path | None:
        if not self.config["video"].get("enabled", True):
            return None
        suffix = preferred_video_suffix()
        video_name = f"{safe_stem(run_name)}{suffix}"
        video_path = self.base_output / run_name / video_name
        synthesize_video(frames, video_path, fps=int(self.config["video"]["fps"]))
        return video_path

    def _predict_attributes(self, word: str, target_layer: int, token_profile: dict[str, Any]) -> dict[str, Any]:
        if token_profile.get("attribute_mode") == "symbolic":
            return self.symbolic_registry.build_predictions(word)
        fitted = self._get_fitted_attribute_probes(target_layer)
        result = predict_word_attributes(self.bundle, fitted, word, target_layer)
        return result["predictions"]

    def _classify_token(self, word: str) -> dict[str, Any]:
        if word in self._token_lists["english_punctuation"]:
            return self.symbolic_registry.get_token_profile(word)
        if word in self._token_lists["chinese_punctuation"]:
            return self.symbolic_registry.get_token_profile(word)
        if word in self._token_lists["arabic_digits"]:
            return self.symbolic_registry.get_token_profile(word)
        if word in self._token_lists["roman_digits"]:
            return self.symbolic_registry.get_token_profile(word)
        if re.fullmatch(r"[A-Za-z_]+", word):
            return {
                "token_class": "lexical_word",
                "attribute_mode": "lexical",
                "match_domain": "lexical",
                "attribute_probe_enabled": True,
                "reason": "Lexical attribute probes are enabled.",
            }
        return {
            "token_class": "other_symbolic_token",
            "attribute_mode": "disabled",
            "match_domain": "none",
            "attribute_probe_enabled": False,
            "reason": "Lexical attribute probes are disabled for symbolic tokens.",
        }

    def _matcher_for_profile(self, token_profile: dict[str, Any]):
        match_domain = token_profile.get("match_domain")
        if match_domain == "symbolic":
            return self.symbolic_matcher
        if match_domain == "lexical":
            return self.matcher
        return None

    def _matcher_for_words(self, words: list[str]):
        symbolic = 0
        lexical = 0
        for word in words:
            profile = self._classify_token(word)
            if profile.get("match_domain") == "symbolic":
                symbolic += 1
            elif profile.get("match_domain") == "lexical":
                lexical += 1
        return self.symbolic_matcher if symbolic > lexical else self.matcher

    def _get_fitted_attribute_probes(self, target_layer: int) -> dict[str, Any]:
        if self._attribute_probe_cache is None:
            rows = load_attribute_rows("data/word_attributes.csv")
            feature_bank = build_feature_bank(self.bundle, rows, target_layer, config=self.config)
            self._attribute_probe_cache = fit_full_attribute_probes(feature_bank, rows)
        return self._attribute_probe_cache

    def _contrast_dim_row(
        self,
        dim: int,
        left_vector: np.ndarray,
        right_vector: np.ndarray,
        diff_vector: np.ndarray,
        mode: str,
        matcher: ConceptMatcher,
    ) -> dict[str, Any]:
        left_value = float(left_vector[dim])
        right_value = float(right_vector[dim])
        diff_value = float(diff_vector[dim])
        shared_strength = float(min(abs(left_value), abs(right_value)))
        diff_strength = float(abs(diff_value))

        left_direction = "positive" if left_value >= 0 else "negative"
        right_direction = "positive" if right_value >= 0 else "negative"
        diff_direction = "positive" if diff_value >= 0 else "negative"
        left_concepts, left_types = matcher.explain_dimension_axis(dim, direction=left_direction, scale=abs(left_value), top_k=3)
        right_concepts, right_types = matcher.explain_dimension_axis(dim, direction=right_direction, scale=abs(right_value), top_k=3)
        diff_concepts, diff_types = matcher.explain_dimension_axis(dim, direction=diff_direction, scale=abs(diff_value), top_k=3)
        return {
            "dim": dim,
            "left_value": left_value,
            "right_value": right_value,
            "diff_value": diff_value,
            "shared_strength": shared_strength,
            "diff_strength": diff_strength,
            "mode": mode,
            "left_top_type": left_types[0]["term"] if left_types else "",
            "left_top_concept": left_concepts[0]["term"] if left_concepts else "",
            "right_top_type": right_types[0]["term"] if right_types else "",
            "right_top_concept": right_concepts[0]["term"] if right_concepts else "",
            "diff_top_type": diff_types[0]["term"] if diff_types else "",
            "diff_top_concept": diff_concepts[0]["term"] if diff_concepts else "",
        }

    @staticmethod
    def _auto_select_salient_dims(vectors: dict[str, np.ndarray], top_k: int) -> list[int]:
        stacked = np.stack(list(vectors.values()), axis=0)
        mean_abs = np.mean(np.abs(stacked), axis=0)
        indices = np.argsort(mean_abs)[::-1][:top_k]
        return [int(idx) for idx in indices]

    @staticmethod
    def _dimension_attribute_group_means(
        dim: int,
        scored: list[dict[str, Any]],
        attribute_map: dict[str, dict[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, dict[str, list[float]]] = {}
        for row in scored:
            word = row["word"]
            attrs = attribute_map.get(word, {})
            value = float(row["value"])
            for attr_name, attr_value in attrs.items():
                if attr_name == "word" or not attr_value:
                    continue
                grouped.setdefault(attr_name, {}).setdefault(attr_value, []).append(value)

        summary: dict[str, list[dict[str, Any]]] = {}
        for attr_name, labels in grouped.items():
            rows = []
            for label, values in labels.items():
                rows.append({"label": label, "mean": float(np.mean(values)), "count": len(values)})
            rows.sort(key=lambda item: abs(item["mean"]), reverse=True)
            summary[attr_name] = rows[:6]
        return summary

    def _resolve_word_files(self, word_files: list[str] | None, word_dir: str, pattern: str) -> list[Path]:
        if word_files:
            return [Path(item) for item in word_files]
        base_dir = Path(word_dir)
        return sorted(path for path in base_dir.glob(pattern) if path.is_file())

    def _build_word_list_profile(self, words: list[str]) -> dict[str, Any]:
        token_counts: Counter[str] = Counter()
        attribute_probe_words = 0
        for word in words:
            token_profile = self._classify_token(word)
            token_counts[token_profile["token_class"]] += 1
            if token_profile["attribute_probe_enabled"]:
                attribute_probe_words += 1
        if len(token_counts) == 1:
            list_profile = next(iter(token_counts))
        elif token_counts:
            list_profile = "mixed"
        else:
            list_profile = "empty"
        token_classes = ", ".join(f"{label}:{count}" for label, count in sorted(token_counts.items()))
        return {
            "attribute_probe_words": attribute_probe_words,
            "list_profile": list_profile,
            "token_classes": token_classes,
        }

    @staticmethod
    def _relative_run_path(path_value: str | None, run_dir: Path) -> str:
        if not path_value:
            return ""
        try:
            return Path(path_value).relative_to(run_dir).as_posix()
        except ValueError:
            return Path(path_value).as_posix()

    def _single_summary_row(self, index: int, result: dict[str, Any]) -> dict[str, Any]:
        attr_predictions = result.get("attribute_predictions") or {}
        token_profile = result.get("token_profile") or {}
        if token_profile.get("match_domain") == "symbolic":
            top_type, top_catalog_type, top_value = self._symbolic_summary_fields(attr_predictions, result)
        else:
            top_type = result["matches"]["types"][0]["term"] if result["matches"]["types"] else ""
            top_value = result["matches"]["values"][0]["term"] if result["matches"]["values"] else ""
            top_catalog_type = result["matches"]["catalog_types"][0]["term"] if result["matches"]["catalog_types"] else ""
        stem = safe_stem(result["word"])
        return {
            "index": index,
            "word": result["word"],
            "target_layer": result["target_layer"],
            "top_type": top_type,
            "top_catalog_type": top_catalog_type,
            "top_value": top_value,
            "pred_category": self._top_attr_label(attr_predictions, "category"),
            "pred_color": self._top_attr_label(attr_predictions, "color"),
            "pred_shape": self._top_attr_label(attr_predictions, "shape"),
            "pred_taste": self._top_attr_label(attr_predictions, "taste"),
            "top_dim": result["top_dims"][0]["dim"] if result["top_dims"] else "",
            "figure": f"figures/{index:04d}_{stem}.png",
            "data": f"data/{index:04d}_{stem}.json",
        }

    def _symbolic_summary_fields(self, attr_predictions: dict[str, Any], result: dict[str, Any]) -> tuple[str, str, str]:
        top_type = self._top_prediction_label(attr_predictions, "token family")
        top_catalog_type = (
            self._top_prediction_label(attr_predictions, "punctuation role")
            or self._top_prediction_label(attr_predictions, "digit family")
            or self._top_prediction_label(attr_predictions, "symbol shape")
        )
        top_value = (
            self._top_prediction_label(attr_predictions, "digit value")
            or self._top_prediction_label(attr_predictions, "symbol name")
            or self._top_prediction_label(attr_predictions, "symbol shape")
        )
        if not top_type:
            top_type = result["matches"]["types"][0]["term"] if result["matches"]["types"] else ""
        if not top_catalog_type:
            top_catalog_type = result["matches"]["catalog_types"][0]["term"] if result["matches"]["catalog_types"] else ""
        if not top_value:
            top_value = result["matches"]["values"][0]["term"] if result["matches"]["values"] else ""
        return top_type, top_catalog_type, top_value

    @staticmethod
    def _top_prediction_label(attr_predictions: dict[str, Any], attr_name: str) -> str:
        payload = attr_predictions.get(attr_name) or {}
        top = payload.get("top_prediction") or {}
        return str(top.get("label", ""))

    def _multi_summary_row(self, index: int, result: dict[str, Any]) -> dict[str, Any]:
        words = result["words"]
        strengths = np.asarray(result["layer_strengths"])
        strongest_layer = int(np.argmax(strengths.mean(axis=1)))
        stem = safe_stem("_".join(words))
        return {
            "index": index,
            "words": " ".join(words),
            "batch_size": len(words),
            "strongest_layer": strongest_layer,
            "figure": f"figures/{index:04d}_{stem}.png",
            "data": f"data/{index:04d}_{stem}.json",
        }

    def _color_dim_summary_row(self, index: int, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": index,
            "input": result["word"],
            "target_layer": result["target_layer"],
            "token_count": len(result.get("tokens", [])),
            "max_dims": self._format_dim_list(result.get("top_max_dims", [])),
            "min_dims": self._format_dim_list(result.get("top_min_dims", [])),
            "max_dim_1": result.get("top_max_dims", [{}])[0].get("dim", "") if result.get("top_max_dims") else "",
            "max_value_1": result.get("top_max_dims", [{}])[0].get("value", "") if result.get("top_max_dims") else "",
            "min_dim_1": result.get("top_min_dims", [{}])[0].get("dim", "") if result.get("top_min_dims") else "",
            "min_value_1": result.get("top_min_dims", [{}])[0].get("value", "") if result.get("top_min_dims") else "",
        }

    def _build_positional_result(
        self,
        word: str,
        tokens: list[str],
        token_positions: list[int],
        last_token_pos: int,
        vector: np.ndarray,
        target_layer: int,
    ) -> dict[str, Any]:
        """Build a result dict for a word extracted from its positional slot in the sequence."""
        top_k = int(self.config["analysis"]["top_k_dims"])
        top_dims = summarize_top_dims(vector, top_k)
        return {
            "word": word,
            "target_layer": target_layer,
            "tokens": tokens,
            "top_dims": top_dims,
            "top_max_dims": self._top_max_dims_abs(vector, top_k=10),
            "top_min_dims": self._top_min_dims_abs(vector, top_k=10),
            "matches": {"types": [], "values": [], "catalog_types": []},
            "dimension_semantics": [],
            "attribute_predictions": None,
            "token_profile": None,
            "positional_info": {
                "token_positions": token_positions,
                "last_token_pos": last_token_pos,
            },
        }

    @staticmethod
    def _all_input_dim_stats_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Build per-dim stat rows for the all-input (single combined) result.

        Since there is only one input, occurrence_count is always 1 and
        mean_abs_value equals max_abs_value equals abs(value).
        """
        rows: list[dict[str, Any]] = []
        for item in result.get("top_dims", []):
            signed_value = float(item["value"])
            abs_value = float(item["abs_value"])
            rows.append(
                {
                    "dim": int(item["dim"]),
                    "signed_value": signed_value,
                    "direction": "positive" if signed_value >= 0 else "negative",
                    "occurrence_count": 1,
                    "abs_value": abs_value,
                    "mean_abs_value": abs_value,
                    "max_abs_value": abs_value,
                }
            )
        rows.sort(key=lambda r: r["abs_value"], reverse=True)
        return rows

    @staticmethod
    def _full_dim_average_rows(vectors: list[np.ndarray]) -> list[dict[str, Any]]:
        """Build full hidden-dimension statistics across all vectors.

        This keeps every dimension in the layer, e.g. all 4096 dims for Llama-3-8B.
        mean_value is the signed arithmetic mean: sum(value_dim_i) / sample_count.
        """
        if not vectors:
            return []
        stacked = np.stack([np.asarray(vector, dtype=np.float32) for vector in vectors], axis=0)
        mean_values = stacked.mean(axis=0)
        mean_abs_values = np.abs(stacked).mean(axis=0)
        max_values = stacked.max(axis=0)
        min_values = stacked.min(axis=0)
        std_values = stacked.std(axis=0)
        abs_max_values = np.abs(stacked).max(axis=0)
        positive_counts = (stacked > 0).sum(axis=0)
        negative_counts = (stacked < 0).sum(axis=0)
        zero_counts = (stacked == 0).sum(axis=0)
        sample_count = int(stacked.shape[0])
        rows: list[dict[str, Any]] = []
        for dim in range(stacked.shape[1]):
            mean_value = float(mean_values[dim])
            rows.append(
                {
                    "dim": dim,
                    "sample_count": sample_count,
                    "mean_value": mean_value,
                    "mean_abs_value": float(mean_abs_values[dim]),
                    "max_value": float(max_values[dim]),
                    "min_value": float(min_values[dim]),
                    "std_value": float(std_values[dim]),
                    "abs_max_value": float(abs_max_values[dim]),
                    "positive_count": int(positive_counts[dim]),
                    "negative_count": int(negative_counts[dim]),
                    "zero_count": int(zero_counts[dim]),
                    "mean_direction": "positive" if mean_value >= 0 else "negative",
                }
            )
        return rows

    @staticmethod
    def _single_vector_full_dim_rows(vector: np.ndarray) -> list[dict[str, Any]]:
        values = np.asarray(vector, dtype=np.float32)
        rows: list[dict[str, Any]] = []
        for dim, value in enumerate(values):
            signed_value = float(value)
            rows.append(
                {
                    "dim": dim,
                    "sample_count": 1,
                    "mean_value": signed_value,
                    "mean_abs_value": abs(signed_value),
                    "max_value": signed_value,
                    "min_value": signed_value,
                    "std_value": 0.0,
                    "abs_max_value": abs(signed_value),
                    "positive_count": 1 if signed_value > 0 else 0,
                    "negative_count": 1 if signed_value < 0 else 0,
                    "zero_count": 1 if signed_value == 0 else 0,
                    "mean_direction": "positive" if signed_value >= 0 else "negative",
                }
            )
        return rows

    @staticmethod
    def _top_max_dims_abs(vector: np.ndarray, top_k: int = 10) -> list[dict[str, float]]:
        indices = np.argsort(vector)[::-1][:top_k]
        return [
            {
                "dim": int(idx),
                "value": float(abs(vector[idx])),
                "signed_value": float(vector[idx]),
                "direction": "positive",
            }
            for idx in indices
        ]

    @staticmethod
    def _top_min_dims_abs(vector: np.ndarray, top_k: int = 10) -> list[dict[str, float]]:
        indices = np.argsort(vector)[:top_k]
        return [
            {
                "dim": int(idx),
                "value": float(abs(vector[idx])),
                "signed_value": float(vector[idx]),
                "direction": "negative",
            }
            for idx in indices
        ]

    @staticmethod
    def _format_dim_list(items: list[dict[str, Any]]) -> str:
        return " | ".join(f"{item.get('dim')}:{float(item.get('value', 0.0)):.6f}" for item in items)

    def _color_dim_average_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int], list[float]] = {}
        for row in rows:
            for group_name, field_name in (("max", "max_dims"), ("min", "min_dims")):
                for item in self._parse_dim_list(str(row.get(field_name, ""))):
                    grouped.setdefault((group_name, item["dim"]), []).append(item["value"])

        average_rows: list[dict[str, Any]] = []
        for (group_name, dim), values in grouped.items():
            values_array = np.asarray(values, dtype=np.float32)
            average_rows.append(
                {
                    "group": group_name,
                    "dim": dim,
                    "appearance_count": int(values_array.size),
                    "mean_abs_value": float(values_array.mean()),
                    "max_abs_value": float(values_array.max()),
                    "min_abs_value": float(values_array.min()),
                }
            )
        average_rows.sort(key=lambda item: (item["group"], -float(item["mean_abs_value"]), int(item["dim"])))
        return average_rows

    @staticmethod
    def _parse_dim_list(value: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in value.split("|"):
            text = part.strip()
            if not text or ":" not in text:
                continue
            dim_text, value_text = text.split(":", 1)
            try:
                items.append({"dim": int(dim_text), "value": float(value_text)})
            except ValueError:
                continue
        return items

    def _write_run_index(self, run_name: str, rows: list[dict[str, Any]], mode: str, video_path: Path | None = None) -> None:
        if not rows:
            return
        run_dir = ensure_dir(self.base_output / run_name)
        video_rel = ""
        if video_path:
            try:
                video_rel = video_path.relative_to(run_dir).as_posix()
            except ValueError:
                video_rel = video_path.as_posix()
        write_csv(run_dir / "summary.csv", rows)
        write_json(run_dir / "summary.json", {"mode": mode, "rows": rows, "video": video_rel})
        write_text(run_dir / "index.html", self._build_html_index(run_name, rows, mode, video_rel))
        if mode == "single":
            overview = self._build_single_overview(rows)
            overview["video"] = video_rel
            write_json(run_dir / "overview.json", overview)
            write_text(run_dir / "overview.html", self._build_single_overview_html(run_name, overview))
        elif mode == "multi":
            overview = self._build_multi_overview(rows)
            overview["video"] = video_rel
            write_json(run_dir / "overview.json", overview)
            write_text(run_dir / "overview.html", self._build_multi_overview_html(run_name, overview))

    def _build_html_index(self, run_name: str, rows: list[dict[str, Any]], mode: str, video_rel: str = "") -> str:
        title = f"{run_name} Results"
        if mode == "single":
            columns = [
                "index",
                "word",
                "top_type",
                "top_catalog_type",
                "top_value",
                "pred_category",
                "pred_color",
                "pred_shape",
                "pred_taste",
                "top_dim",
                "figure",
                "data",
            ]
        else:
            columns = ["index", "words", "batch_size", "strongest_layer", "figure", "data"]

        header_cells = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body_rows: list[str] = []
        for row in rows:
            attrs = ""
            if mode == "single":
                category_label, _ = self._parse_attr_label_and_score(str(row.get("pred_category", "")))
                color_label, _ = self._parse_attr_label_and_score(str(row.get("pred_color", "")))
                shape_label, _ = self._parse_attr_label_and_score(str(row.get("pred_shape", "")))
                taste_label, _ = self._parse_attr_label_and_score(str(row.get("pred_taste", "")))
                attrs = (
                    f" data-word=\"{html.escape(str(row.get('word', '')).lower())}\""
                    f" data-category=\"{html.escape(category_label.lower())}\""
                    f" data-color=\"{html.escape(color_label.lower())}\""
                    f" data-shape=\"{html.escape(shape_label.lower())}\""
                    f" data-taste=\"{html.escape(taste_label.lower())}\""
                    f" data-category-score=\"{self._extract_attr_score(str(row.get('pred_category', '')))}\""
                    f" data-color-score=\"{self._extract_attr_score(str(row.get('pred_color', '')))}\""
                    f" data-shape-score=\"{self._extract_attr_score(str(row.get('pred_shape', '')))}\""
                    f" data-taste-score=\"{self._extract_attr_score(str(row.get('pred_taste', '')))}\""
                )
            cells = []
            for column in columns:
                value = row.get(column, "")
                if column in {"figure", "data"} and value:
                    cell = f'<a href="{html.escape(str(value))}">{html.escape(str(value))}</a>'
                else:
                    cell = html.escape(str(value))
                cells.append(f"<td>{cell}</td>")
            body_rows.append(f"<tr{attrs}>{''.join(cells)}</tr>")

        filter_bar = ""
        filter_script = ""
        if mode == "single":
            filter_bar = (
                "<div class='filters'>"
                "<input id='filter-word' placeholder='Filter word' />"
                "<input id='filter-category' placeholder='Filter category' />"
                "<input id='filter-color' placeholder='Filter color' />"
                "<input id='filter-shape' placeholder='Filter shape' />"
                "<input id='filter-taste' placeholder='Filter taste' />"
                "<input id='filter-min-score' placeholder='Min confidence (0-1)' />"
                "<label class='check'><input type='checkbox' id='filter-any-high' /> any core attr above threshold</label>"
                "</div>"
            )
            filter_script = """
  <script>
    const textInputs = ['word', 'category', 'color', 'shape', 'taste'].map((key) => document.getElementById('filter-' + key));
    const minScoreInput = document.getElementById('filter-min-score');
    const anyHighInput = document.getElementById('filter-any-high');
    const rows = [...document.querySelectorAll('tbody tr')];
    function attrScore(row, key) {
      const raw = row.dataset[key + 'Score'];
      const num = parseFloat(raw || '0');
      return Number.isFinite(num) ? num : 0;
    }
    function applyFilters() {
      const minScore = parseFloat((minScoreInput && minScoreInput.value) || '0');
      rows.forEach((row) => {
        const textOk = textInputs.every((input) => {
          if (!input || !input.value.trim()) {
            return true;
          }
          const key = input.id.replace('filter-', '');
          const value = (row.dataset[key] || '').toLowerCase();
          return value.includes(input.value.trim().toLowerCase());
        });
        let scoreOk = true;
        if (Number.isFinite(minScore) && minScore > 0) {
          const scores = ['category', 'color', 'shape', 'taste'].map((key) => attrScore(row, key));
          scoreOk = anyHighInput && anyHighInput.checked ? scores.some((value) => value >= minScore) : scores.every((value) => value === 0 || value >= minScore);
        }
        row.style.display = textOk && scoreOk ? '' : 'none';
      });
    }
    [...textInputs, minScoreInput, anyHighInput].forEach((input) => input && input.addEventListener('input', applyFilters));
    if (anyHighInput) {
      anyHighInput.addEventListener('change', applyFilters);
    }
  </script>
"""

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e5e7eb; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }}
    .filters input {{ padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 8px; }}
    .check {{ display: inline-flex; align-items: center; gap: 6px; color: #475569; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Mode: {mode}</p>
  <p>
    {"<a href='overview.html'>Open overview</a>" if mode in {'single', 'multi'} else ""}
    {" | " if mode in {'single', 'multi'} and video_rel else ""}
    {f"<a href='{video_rel}'>Open recording</a>" if video_rel else ""}
  </p>
  {filter_bar}
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
{filter_script}
</body>
</html>
"""

    @staticmethod
    def _build_color_experiment_html(
        run_name: str,
        word_count: int,
        per_word_video: str,
        all_input_video: str,
        positional_video: str,
    ) -> str:
        per_word_video_link = f'<a href="{html.escape(per_word_video)}">Open per-word recording</a>' if per_word_video else "No per-word recording"
        all_input_video_link = f'<a href="{html.escape(all_input_video)}">Open all-input recording</a>' if all_input_video else "No all-input recording"
        positional_video_link = f'<a href="{html.escape(positional_video)}">Open positional recording</a>' if positional_video else "No positional recording"
        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(run_name)} Color Word Experiment</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 12px; padding: 18px; }}
    .full {{ grid-column: 1 / -1; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    img.chart {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>{html.escape(run_name)} Color Word Experiment</h1>
  <p>Word count: {word_count} | Probe layer: 8 | Values: actual signed activations (no normalization)</p>
  <div class=\"grid\">
    <section class=\"card\">
      <h2>Mode 1 — Per-Word (孤立输入)</h2>
      <p>{per_word_video_link}</p>
      <p><a href=\"single_words/index.html\">Open image index</a></p>
      <p><a href=\"per_word_dim_extremes.csv\">Max/min dim CSV</a></p>
      <p><a href=\"per_word_dim_average_abs.csv\">Dimension statistics CSV</a></p>
      <p><a href=\"per_word_full_4096_dim_stats.csv\">Full 4096-dim mean CSV</a></p>
    </section>
    <section class=\"card\">
      <h2>Mode 2 — All-Input (末位 token)</h2>
      <p>{all_input_video_link}</p>
      <p><a href=\"all_input/index.html\">Open image index</a></p>
      <p><a href=\"all_input_dim_extremes.csv\">Max/min dim CSV</a></p>
      <p><a href=\"all_input_dim_stats.csv\">Dimension statistics CSV</a></p>
      <p><a href=\"all_input_full_4096_dim_stats.csv\">Full 4096-dim CSV</a></p>
    </section>
    <section class=\"card\">
      <h2>Mode 3 — Positional (序列内位置提取)</h2>
      <p>{positional_video_link}</p>
      <p><a href=\"positional_words/index.html\">Open image index</a></p>
      <p><a href=\"positional_dim_extremes.csv\">Max/min dim CSV</a></p>
      <p><a href=\"positional_dim_average_abs.csv\">Dimension statistics CSV</a></p>
      <p><a href=\"positional_full_4096_dim_stats.csv\">Full 4096-dim mean CSV</a></p>
    </section>
    <section class=\"card full\">
      <h2>Dimension Statistics — Per-Word Mode</h2>
      <p>Top 20 dims by occurrence count. Bar = mean |activation| (raw). Warm = positive, cool = negative.</p>
      <img class=\"chart\" src=\"per_word_dim_stats_chart.png\" alt=\"Per-word dimension statistics\" />
    </section>
    <section class=\"card full\">
      <h2>Dimension Statistics — All-Input Mode</h2>
      <p>Top dims for the combined all-words input. Bars = actual signed values. No normalization.</p>
      <img class=\"chart\" src=\"all_input_dim_stats_chart.png\" alt=\"All-input dimension statistics\" />
    </section>
    <section class=\"card full\">
      <h2>Dimension Statistics — Positional Mode</h2>
      <p>Top 20 dims by occurrence count across all positional extractions (one forward pass). Bar = mean |activation|.</p>
      <img class=\"chart\" src=\"positional_dim_stats_chart.png\" alt=\"Positional dimension statistics\" />
    </section>
    <section class=\"card full\">
      <h2>Three-Mode Comparison</h2>
      <p>Left: per-word | Centre: all-input | Right: positional.  All charts use raw signed activation values.</p>
      <img class=\"chart\" src=\"three_mode_comparison_chart.png\" alt=\"Three-mode comparison chart\" />
    </section>
    <section class=\"card full\">
      <h2>Full 4096-Dim Mean Landscape</h2>
      <p>All hidden dimensions are shown. Y = raw signed mean_value. Warm fill = positive, cool fill = negative.</p>
      <img class=\"chart\" src=\"full_4096_mean_landscape.png\" alt=\"Full 4096-dim mean landscape\" />
    </section>
    <section class=\"card full\">
      <h2>Full 4096-Dim Top Signed Bars</h2>
      <p>Top dimensions selected by |mean_value| from the complete 4096-dim statistics, not only per-input Top 10.</p>
      <img class=\"chart\" src=\"full_4096_top_signed_bars.png\" alt=\"Full 4096-dim top signed bars\" />
    </section>
  </div>
</body>
</html>
"""

    def _write_global_analysis_bundle(
        self,
        run_name: str,
        rows: list[dict[str, Any]],
        batch_size: int,
        word_dir: str,
        pattern: str,
    ) -> None:
        run_dir = ensure_dir(self.base_output / run_name)
        overview = self._build_global_analysis_overview(rows, batch_size=batch_size, word_dir=word_dir, pattern=pattern)
        write_csv(run_dir / "summary.csv", rows)
        write_json(run_dir / "summary.json", {"rows": rows})
        write_json(run_dir / "overview.json", overview)
        write_text(run_dir / "index.html", self._build_global_analysis_index_html(run_name, rows))
        write_text(run_dir / "overview.html", self._build_global_analysis_overview_html(run_name, overview))

    @staticmethod
    def _build_global_analysis_overview(
        rows: list[dict[str, Any]],
        batch_size: int,
        word_dir: str,
        pattern: str,
    ) -> dict[str, Any]:
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        failed_rows = [row for row in rows if row.get("status") != "ok"]
        total_tokens = sum(int(row.get("word_count", 0) or 0) for row in rows)
        total_probe_words = sum(int(row.get("attribute_probe_words", 0) or 0) for row in rows)
        available_videos = sum(1 for row in rows if row.get("single_video") or row.get("multi_video"))
        return {
            "word_dir": word_dir,
            "pattern": pattern,
            "batch_size": batch_size,
            "total_lists": len(rows),
            "successful_lists": len(ok_rows),
            "failed_lists": len(failed_rows),
            "total_tokens": total_tokens,
            "total_attribute_probe_words": total_probe_words,
            "lists_with_video": available_videos,
            "list_profiles": ProbePipeline._count_ranked(rows, "list_profile"),
            "largest_lists": ProbePipeline._top_global_rows(rows, sort_key="word_count", limit=8),
            "probe_heavy_lists": ProbePipeline._top_global_rows(rows, sort_key="attribute_probe_words", limit=8),
            "failures": [
                {"list_name": row.get("list_name", ""), "word_file": row.get("word_file", ""), "error": row.get("error", "")}
                for row in failed_rows
            ],
        }

    @staticmethod
    def _top_global_rows(rows: list[dict[str, Any]], sort_key: str, limit: int = 8) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for row in rows:
            try:
                score = int(row.get(sort_key, 0) or 0)
            except (TypeError, ValueError):
                score = 0
            ranked.append(
                {
                    "list_name": row.get("list_name", ""),
                    "word_count": row.get("word_count", 0),
                    "attribute_probe_words": row.get("attribute_probe_words", 0),
                    "list_profile": row.get("list_profile", ""),
                    "status": row.get("status", ""),
                    "score": score,
                    "single_overview": row.get("single_overview", ""),
                    "multi_overview": row.get("multi_overview", ""),
                    "dimension_report": row.get("dimension_report", ""),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def _build_global_analysis_index_html(self, run_name: str, rows: list[dict[str, Any]]) -> str:
        body_rows = []
        for row in rows:
            links = []
            for label, key in (
                ("single index", "single_index"),
                ("single overview", "single_overview"),
                ("single video", "single_video"),
                ("multi index", "multi_index"),
                ("multi overview", "multi_overview"),
                ("multi video", "multi_video"),
                ("dim report", "dimension_report"),
            ):
                value = str(row.get(key, ""))
                if value:
                    links.append(f'<a href="{html.escape(value)}">{label}</a>')
            body_rows.append(
                "<tr>"
                f"<td>{row.get('index', '')}</td>"
                f"<td>{html.escape(str(row.get('list_name', '')))}</td>"
                f"<td>{html.escape(str(row.get('word_file', '')))}</td>"
                f"<td>{row.get('word_count', '')}</td>"
                f"<td>{row.get('attribute_probe_words', '')}</td>"
                f"<td>{html.escape(str(row.get('list_profile', '')))}</td>"
                f"<td>{html.escape(str(row.get('token_classes', '')))}</td>"
                f"<td>{html.escape(str(row.get('sample_words', '')))}</td>"
                f"<td>{html.escape(str(row.get('status', '')))}</td>"
                f"<td>{html.escape(str(row.get('error', '')))}</td>"
                f"<td>{' | '.join(links) if links else ''}</td>"
                "</tr>"
            )
        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{run_name} Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e5e7eb; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{run_name} Index</h1>
  <p><a href=\"overview.html\">Open aggregate overview</a> | Each list row includes single/multi recordings when available.</p>
  <table>
    <thead>
      <tr>
        <th>index</th><th>list_name</th><th>word_file</th><th>word_count</th><th>attribute_probe_words</th><th>list_profile</th><th>token_classes</th><th>sample_words</th><th>status</th><th>error</th><th>links</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""

    def _build_global_analysis_overview_html(self, run_name: str, overview: dict[str, Any]) -> str:
        def render_list(items: list[dict[str, Any]], key: str) -> str:
            if not items:
                return "<li>None</li>"
            return "".join(
                f"<li><strong>{html.escape(str(item['list_name']))}</strong>: {key}={item['score']} | profile={html.escape(str(item['list_profile']))}</li>"
                for item in items
            )

        def render_profile_counts(items: list[dict[str, Any]]) -> str:
            if not items:
                return "<li>None</li>"
            return "".join(f"<li><strong>{html.escape(str(item['label']))}</strong>: {item['count']}</li>" for item in items)

        def render_failures(items: list[dict[str, Any]]) -> str:
            if not items:
                return "<li>None</li>"
            return "".join(
                f"<li><strong>{html.escape(str(item['list_name']))}</strong>: {html.escape(str(item['error']))}</li>"
                for item in items
            )

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{run_name} Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 16px; padding: 18px; }}
    h1, h2 {{ margin-top: 0; }}
    ul {{ padding-left: 20px; }}
    .meta {{ color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{run_name} Overview</h1>
  <p class=\"meta\">Lists: {overview['total_lists']} | Success: {overview['successful_lists']} | Failed: {overview['failed_lists']} | Tokens: {overview['total_tokens']} | Attr-probe words: {overview['total_attribute_probe_words']} | Lists with recordings: {overview['lists_with_video']} | <a href=\"index.html\">Open aggregate index</a></p>
  <p class=\"meta\">Discovery: {html.escape(str(overview['word_dir']))}/{html.escape(str(overview['pattern']))} | multi batch size = {overview['batch_size']}</p>
  <div class=\"grid\">
    <section class=\"card\"><h2>List Profiles</h2><ul>{render_profile_counts(overview['list_profiles'])}</ul></section>
    <section class=\"card\"><h2>Largest Lists</h2><ul>{render_list(overview['largest_lists'], 'word_count')}</ul></section>
    <section class=\"card\"><h2>Probe-heavy Lists</h2><ul>{render_list(overview['probe_heavy_lists'], 'attribute_probe_words')}</ul></section>
    <section class=\"card\"><h2>Failures</h2><ul>{render_failures(overview['failures'])}</ul></section>
  </div>
</body>
</html>
"""

    @staticmethod
    def _top_attr_label(predictions: dict[str, Any], attribute: str) -> str:
        payload = predictions.get(attribute, {})
        top = payload.get("top_prediction") or {}
        label = top.get("label", "")
        score = top.get("score")
        if not label:
            return ""
        if score is None:
            return str(label)
        return f"{label} ({score:.3f})"

    @staticmethod
    def _extract_attr_score(text: str) -> str:
        if "(" not in text or ")" not in text:
            return "0"
        try:
            return f"{float(text.split('(')[-1].split(')')[0]):.3f}"
        except ValueError:
            return "0"

    @staticmethod
    def _parse_attr_label_and_score(text: str) -> tuple[str, float]:
        if not text:
            return "", 0.0
        if "(" not in text or not text.endswith(")"):
            return text.strip(), 0.0
        label = text.rsplit("(", 1)[0].strip()
        try:
            score = float(text.rsplit("(", 1)[1][:-1])
        except ValueError:
            score = 0.0
        return label, score

    def _build_single_overview(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_words": len(rows),
            "top_categories": self._count_ranked(rows, "pred_category"),
            "top_shapes": self._count_ranked(rows, "pred_shape"),
            "top_colors": self._count_ranked(rows, "pred_color"),
            "top_tastes": self._count_ranked(rows, "pred_taste"),
            "highest_confidence": ProbePipeline._highest_confidence_rows(rows),
            "review_candidates": ProbePipeline._review_candidate_rows(rows),
        }

    @staticmethod
    def _count_ranked(rows: list[dict[str, Any]], key: str, limit: int = 8) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in rows:
            label, _score = ProbePipeline._parse_attr_label_and_score(str(row.get(key, "")))
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        ranked = [{"label": label, "count": count} for label, count in counts.items()]
        ranked.sort(key=lambda item: (-item["count"], item["label"]))
        return ranked[:limit]

    @staticmethod
    def _highest_confidence_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for row in rows:
            for attr in ("pred_category", "pred_color", "pred_shape", "pred_taste"):
                label, score = ProbePipeline._parse_attr_label_and_score(str(row.get(attr, "")))
                if not label:
                    continue
                scored.append(
                    {
                        "word": row.get("word", ""),
                        "attribute": attr.replace("pred_", ""),
                        "label": label,
                        "score": score,
                        "figure": row.get("figure", ""),
                        "data": row.get("data", ""),
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _review_candidate_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            scores = []
            labels = {}
            for attr in ("pred_category", "pred_color", "pred_shape", "pred_taste"):
                label, score = ProbePipeline._parse_attr_label_and_score(str(row.get(attr, "")))
                if label:
                    scores.append(score)
                    labels[attr.replace("pred_", "")] = label
            if not scores:
                continue
            mean_score = sum(scores) / len(scores)
            if mean_score <= 0.55:
                candidates.append(
                    {
                        "word": row.get("word", ""),
                        "mean_score": round(mean_score, 3),
                        "labels": labels,
                        "figure": row.get("figure", ""),
                        "data": row.get("data", ""),
                    }
                )
        candidates.sort(key=lambda item: item["mean_score"])
        return candidates[:limit]

    def _build_single_overview_html(self, run_name: str, overview: dict[str, Any]) -> str:
        def render_list(items: list[dict[str, Any]], label_key: str, value_key: str) -> str:
            if not items:
                return "<li>None</li>"
            return "".join(f"<li><strong>{item[label_key]}</strong>: {item[value_key]}</li>" for item in items)

        def render_high_conf(items: list[dict[str, Any]]) -> str:
            if not items:
                return "<li>None</li>"
            rows = []
            for item in items:
                link = f"<a href='{item['figure']}'>figure</a> | <a href='{item['data']}'>data</a>"
                rows.append(f"<li><strong>{item['word']}</strong> {item['attribute']} = {item['label']} ({item['score']:.3f}) [{link}]</li>")
            return "".join(rows)

        def render_review(items: list[dict[str, Any]]) -> str:
            if not items:
                return "<li>None</li>"
            rows = []
            for item in items:
                labels = ", ".join(f"{key}:{value}" for key, value in item["labels"].items())
                link = f"<a href='{item['figure']}'>figure</a> | <a href='{item['data']}'>data</a>"
                rows.append(f"<li><strong>{item['word']}</strong> mean={item['mean_score']:.3f} | {labels} [{link}]</li>")
            return "".join(rows)

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{run_name} Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 16px; padding: 18px; }}
    h1, h2 {{ margin-top: 0; }}
    ul {{ padding-left: 20px; }}
    .meta {{ color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{run_name} Overview</h1>
  <p class=\"meta\">Total words: {overview['total_words']} | <a href=\"index.html\">Open batch index</a>{f" | <a href='{overview.get('video', '')}'>Open recording</a>" if overview.get('video') else ""}</p>
  <div class=\"grid\">
    <section class=\"card\"><h2>Top Categories</h2><ul>{render_list(overview['top_categories'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Top Shapes</h2><ul>{render_list(overview['top_shapes'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Top Colors</h2><ul>{render_list(overview['top_colors'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Top Tastes</h2><ul>{render_list(overview['top_tastes'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Highest Confidence Predictions</h2><ul>{render_high_conf(overview['highest_confidence'])}</ul></section>
    <section class=\"card\"><h2>Review Candidates</h2><ul>{render_review(overview['review_candidates'])}</ul></section>
  </div>
</body>
</html>
"""

    def _build_multi_overview(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_batches": len(rows),
            "batch_sizes": self._count_ranked(rows, "batch_size"),
            "strongest_layers": self._count_ranked(rows, "strongest_layer"),
            "top_batches": self._top_multi_batches(rows),
            "review_candidates": self._multi_review_candidates(rows),
        }

    @staticmethod
    def _top_multi_batches(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        ranked = []
        for row in rows:
            try:
                strongest_layer = int(row.get("strongest_layer", 0))
            except (TypeError, ValueError):
                strongest_layer = 0
            ranked.append(
                {
                    "words": row.get("words", ""),
                    "batch_size": row.get("batch_size", ""),
                    "strongest_layer": strongest_layer,
                    "figure": row.get("figure", ""),
                    "data": row.get("data", ""),
                }
            )
        ranked.sort(key=lambda item: item["strongest_layer"], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _multi_review_candidates(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        ranked = []
        for row in rows:
            try:
                strongest_layer = int(row.get("strongest_layer", 0))
            except (TypeError, ValueError):
                strongest_layer = 0
            if strongest_layer <= 0:
                continue
            ranked.append(
                {
                    "words": row.get("words", ""),
                    "batch_size": row.get("batch_size", ""),
                    "strongest_layer": strongest_layer,
                    "figure": row.get("figure", ""),
                    "data": row.get("data", ""),
                }
            )
        ranked.sort(key=lambda item: item["strongest_layer"])
        return ranked[:limit]

    def _build_multi_overview_html(self, run_name: str, overview: dict[str, Any]) -> str:
        def render_list(items: list[dict[str, Any]], label_key: str, value_key: str) -> str:
            if not items:
                return "<li>None</li>"
            return "".join(f"<li><strong>{item[label_key]}</strong>: {item[value_key]}</li>" for item in items)

        def render_batches(items: list[dict[str, Any]]) -> str:
            if not items:
                return "<li>None</li>"
            rows = []
            for item in items:
                link = f"<a href='{item['figure']}'>figure</a> | <a href='{item['data']}'>data</a>"
                rows.append(f"<li><strong>{item['words']}</strong> layer={item['strongest_layer']} batch={item['batch_size']} [{link}]</li>")
            return "".join(rows)

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{run_name} Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 16px; padding: 18px; }}
    h1, h2 {{ margin-top: 0; }}
    ul {{ padding-left: 20px; }}
    .meta {{ color: #4b5563; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{run_name} Overview</h1>
  <p class=\"meta\">Total batches: {overview['total_batches']} | <a href=\"index.html\">Open batch index</a>{f" | <a href='{overview.get('video', '')}'>Open recording</a>" if overview.get('video') else ""}</p>
  <div class=\"grid\">
    <section class=\"card\"><h2>Batch Sizes</h2><ul>{render_list(overview['batch_sizes'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Strongest Layers</h2><ul>{render_list(overview['strongest_layers'], 'label', 'count')}</ul></section>
    <section class=\"card\"><h2>Top Batches</h2><ul>{render_batches(overview['top_batches'])}</ul></section>
    <section class=\"card\"><h2>Review Candidates</h2><ul>{render_batches(overview['review_candidates'])}</ul></section>
  </div>
</body>
</html>
"""

    def _build_dimension_report_html(self, report: dict[str, Any]) -> str:
        groups = self._build_dimension_groups(report)
        network_svg = self._build_dimension_network_svg(report)
        group_items = "".join(
            f"<li><strong>{html.escape(item['label'])}</strong>: dims={', '.join(str(dim) for dim in item['dims'])}</li>"
            for item in groups
        ) or "<li>None</li>"
        cards: list[str] = []
        for item in report["dims"]:
            pos_words = "".join(f"<li><strong>{html.escape(row['word'])}</strong>: |{abs(row['value']):.3f}|</li>" for row in item["top_positive_words"])
            neg_words = "".join(f"<li><strong>{html.escape(row['word'])}</strong>: |{abs(row['value']):.3f}|</li>" for row in item["top_negative_words"])
            pos_axis = "".join(f"<li><strong>{html.escape(row['term'])}</strong>: |{abs(row['score']):.3f}|</li>" for row in item["positive_axis"]["types"])
            neg_axis = "".join(f"<li><strong>{html.escape(row['term'])}</strong>: |{abs(row['score']):.3f}|</li>" for row in item["negative_axis"]["types"])
            attr_blocks = []
            for attr_name, attr_rows in item["attribute_group_means"].items():
                entries = "".join(
                    f"<li><strong>{html.escape(row['label'])}</strong>: |mean|={abs(row['mean']):.3f}, n={row['count']}</li>"
                    for row in attr_rows
                )
                attr_blocks.append(f"<div class='subcard'><h4>{html.escape(attr_name)}</h4><ul>{entries}</ul></div>")
            cards.append(
                f"""
                <section class=\"card\">
                  <h2>Dimension {item['dim']}</h2>
                  <div class=\"grid\">
                    <div class=\"subcard\"><h3>Top Positive Words</h3><ul>{pos_words}</ul></div>
                    <div class=\"subcard\"><h3>Top Negative Words</h3><ul>{neg_words}</ul></div>
                    <div class=\"subcard\"><h3>Positive Axis Types</h3><ul>{pos_axis}</ul></div>
                    <div class=\"subcard\"><h3>Negative Axis Types</h3><ul>{neg_axis}</ul></div>
                    {''.join(attr_blocks)}
                  </div>
                </section>
                """
            )

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Dimension Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 16px; padding: 18px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 14px; }}
    .subcard {{ background: #fffdf8; border: 1px solid #e7e5e4; border-radius: 12px; padding: 12px; }}
    h1, h2, h3, h4 {{ margin-top: 0; }}
    ul {{ padding-left: 20px; margin-bottom: 0; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Dimension Stability Report</h1>
  <p class=\"meta\">Layer {report['target_layer']} | words={report['word_count']}</p>
  <section class=\"card\"><h2>Dimension Groups</h2><p class=\"meta\">Grouped by dominant positive-type / negative-type direction.</p><ul>{group_items}</ul></section>
  <section class=\"card\"><h2>Dimension -> Word -> Attribute Network</h2><div class=\"network\">{network_svg}</div></section>
  {''.join(cards)}
</body>
</html>
"""

    @staticmethod
    def _build_dimension_groups(report: dict[str, Any]) -> list[dict[str, Any]]:
        groups: dict[str, list[int]] = {}
        for item in report.get("dims", []):
            pos = item.get("positive_axis", {}).get("types", [])
            neg = item.get("negative_axis", {}).get("types", [])
            pos_label = pos[0]["term"] if pos else "none"
            neg_label = neg[0]["term"] if neg else "none"
            label = f"{pos_label} / {neg_label}"
            groups.setdefault(label, []).append(int(item["dim"]))
        rows = [{"label": label, "dims": dims, "count": len(dims)} for label, dims in groups.items()]
        rows.sort(key=lambda item: (-item["count"], item["label"]))
        return rows

    @staticmethod
    def _build_dimension_network_svg(report: dict[str, Any]) -> str:
        dims = report.get("dims", [])[:16]
        if not dims:
            return "<p>No network data.</p>"

        word_names: list[str] = []
        attr_names: list[str] = []
        for item in dims:
            for row in item.get("top_positive_words", [])[:2]:
                if row["word"] not in word_names:
                    word_names.append(row["word"])
            for row in item.get("top_negative_words", [])[:2]:
                if row["word"] not in word_names:
                    word_names.append(row["word"])
            for attr_name, attr_rows in item.get("attribute_group_means", {}).items():
                for row in attr_rows[:1]:
                    name = f"{attr_name}:{row['label']}"
                    if name not in attr_names:
                        attr_names.append(name)

        word_names = word_names[:18]
        attr_names = attr_names[:14]
        width = 1120
        height = max(560, 90 + max(len(dims), len(word_names), len(attr_names)) * 34)
        x_dim, x_word, x_attr = 140, 530, 930

        def spread(count: int, top: int, bottom: int) -> list[float]:
            if count <= 1:
                return [(top + bottom) / 2]
            return list(np.linspace(top, bottom, count))

        dim_y = spread(len(dims), 70, height - 50)
        word_y = spread(len(word_names), 70, height - 50)
        attr_y = spread(len(attr_names), 70, height - 50)
        word_pos = {name: word_y[idx] for idx, name in enumerate(word_names)}
        attr_pos = {name: attr_y[idx] for idx, name in enumerate(attr_names)}

        parts = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf8" rx="18" />',
            f'<text x="{x_dim}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">dimensions</text>',
            f'<text x="{x_word}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">words</text>',
            f'<text x="{x_attr}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">attribute groups</text>',
        ]

        for idx, item in enumerate(dims):
            y = dim_y[idx]
            dim_label = f"d{item['dim']}"
            parts.append(f'<circle cx="{x_dim}" cy="{y:.1f}" r="18" fill="#3d405b" />')
            parts.append(f'<text x="{x_dim}" y="{y + 5:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="white">{html.escape(dim_label)}</text>')

            for row in item.get("top_positive_words", [])[:2]:
                if row["word"] not in word_pos:
                    continue
                wy = word_pos[row["word"]]
                width_px = 1.2 + 5.0 * min(abs(float(row["value"])), 2.0) / 2.0
                parts.append(f'<line x1="{x_dim + 18}" y1="{y:.1f}" x2="{x_word - 65}" y2="{wy:.1f}" stroke="#e07a5f" stroke-width="{width_px:.2f}" stroke-opacity="0.45" />')

            for row in item.get("top_negative_words", [])[:2]:
                if row["word"] not in word_pos:
                    continue
                wy = word_pos[row["word"]]
                width_px = 1.2 + 5.0 * min(abs(float(row["value"])), 2.0) / 2.0
                parts.append(f'<line x1="{x_dim + 18}" y1="{y:.1f}" x2="{x_word - 65}" y2="{wy:.1f}" stroke="#3d405b" stroke-width="{width_px:.2f}" stroke-opacity="0.35" />')

            for attr_name, attr_rows in item.get("attribute_group_means", {}).items():
                for row in attr_rows[:1]:
                    name = f"{attr_name}:{row['label']}"
                    if name not in attr_pos:
                        continue
                    ay = attr_pos[name]
                    width_px = 1.0 + 4.0 * min(abs(float(row["mean"])), 2.0) / 2.0
                    parts.append(f'<line x1="{x_dim + 18}" y1="{y:.1f}" x2="{x_attr - 90}" y2="{ay:.1f}" stroke="#2a9d8f" stroke-width="{width_px:.2f}" stroke-opacity="0.22" />')

        for name, y in word_pos.items():
            parts.append(f'<rect x="{x_word - 65}" y="{y - 13:.1f}" width="130" height="26" rx="12" fill="#81b29a" />')
            parts.append(f'<text x="{x_word}" y="{y + 5:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="white">{html.escape(name)}</text>')

        for name, y in attr_pos.items():
            parts.append(f'<rect x="{x_attr - 90}" y="{y - 13:.1f}" width="180" height="26" rx="12" fill="#f2cc8f" stroke="#9c6644" stroke-width="1" />')
            parts.append(f'<text x="{x_attr}" y="{y + 5:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#111827">{html.escape(name)}</text>')

        parts.append("</svg>")
        return "".join(parts)

    def _build_word_contrast_html(self, report: dict[str, Any]) -> str:
        def render_rows(rows: list[dict[str, Any]], score_key: str) -> str:
            items = []
            for row in rows:
                items.append(
                    "<tr>"
                    f"<td>{row['dim']}</td>"
                    f"<td>{abs(row['left_value']):.3f}</td>"
                    f"<td>{abs(row['right_value']):.3f}</td>"
                    f"<td>{abs(row['diff_value']):.3f}</td>"
                    f"<td>{abs(row[score_key]):.3f}</td>"
                    f"<td>{html.escape(row['left_top_type'])} / {html.escape(row['left_top_concept'])}</td>"
                    f"<td>{html.escape(row['right_top_type'])} / {html.escape(row['right_top_concept'])}</td>"
                    f"<td>{html.escape(row['diff_top_type'])} / {html.escape(row['diff_top_concept'])}</td>"
                    "</tr>"
                )
            return "".join(items) or "<tr><td colspan='8'>None</td></tr>"

        left = html.escape(report["left_word"])
        right = html.escape(report["right_word"])
        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{left} vs {right} Contrast Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 16px; padding: 18px; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e5e7eb; }}
    h1, h2 {{ margin-top: 0; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>{left} vs {right} Contrast Report</h1>
  <p class=\"meta\">Layer {report['target_layer']} | values shown as absolute magnitudes.</p>
  <section class=\"card\">
    <h2>Shared Dimensions</h2>
    <table>
      <thead><tr><th>dim</th><th>{left}</th><th>{right}</th><th>|diff|</th><th>|shared|</th><th>{left} concept</th><th>{right} concept</th><th>diff concept</th></tr></thead>
      <tbody>{render_rows(report['shared_dims'], 'shared_strength')}</tbody>
    </table>
  </section>
  <section class=\"card\">
    <h2>Difference Dimensions</h2>
    <table>
      <thead><tr><th>dim</th><th>{left}</th><th>{right}</th><th>|diff|</th><th>|diff score|</th><th>{left} concept</th><th>{right} concept</th><th>diff concept</th></tr></thead>
      <tbody>{render_rows(report['diff_dims'], 'diff_strength')}</tbody>
    </table>
  </section>
</body>
</html>
"""
