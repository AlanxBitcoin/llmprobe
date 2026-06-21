from __future__ import annotations

# Responsibility split:
# - This file is the study CLI adapter aggregator.
# - Each study_*.py owns its own register_cli()/try_execute_cli() implementation.
# - This module only wires those per-study helpers together.

import argparse
from typing import Any, Callable

from .study_attribute_group_neurons import register_cli as register_attribute_group_neurons_cli
from .study_attribute_group_neurons import try_execute_cli as try_execute_attribute_group_neurons_cli
from .study_chat_attention_word_replacement import register_cli as register_chat_attention_word_replacement_cli
from .study_chat_attention_word_replacement import try_execute_cli as try_execute_chat_attention_word_replacement_cli
from .study_layer_ffn_neuron_logits_table import register_cli as register_layer_ffn_neuron_logits_table_cli
from .study_layer_ffn_neuron_logits_table import try_execute_cli as try_execute_layer_ffn_neuron_logits_table_cli
from .study_layer_neuron_logits_table import register_cli as register_layer_neuron_logits_table_cli
from .study_layer_neuron_logits_table import try_execute_cli as try_execute_layer_neuron_logits_table_cli
from .study_layer_neurons import register_cli as register_layer_neurons_cli
from .study_layer_neurons import try_execute_cli as try_execute_layer_neurons_cli
from .study_layer_shortcut import register_cli as register_layer_shortcut_cli
from .study_layer_shortcut import try_execute_cli as try_execute_layer_shortcut_cli
from .study_one_on_one_attention import register_cli as register_one_on_one_attention_cli
from .study_one_on_one_attention import try_execute_cli as try_execute_one_on_one_attention_cli
from .study_qk_params import register_cli as register_qk_params_cli
from .study_qk_params import try_execute_cli as try_execute_qk_params_cli
from .study_sentence_next_word import register_cli as register_sentence_next_word_cli
from .study_sentence_next_word import try_execute_cli as try_execute_sentence_next_word_cli
from .study_single_word_hidden_state import register_cli as register_single_word_hidden_state_cli
from .study_single_word_hidden_state import try_execute_cli as try_execute_single_word_hidden_state_cli
from .study_single_word_hidden_state_batch_average import register_cli as register_single_word_hidden_state_batch_average_cli
from .study_single_word_hidden_state_batch_average import try_execute_cli as try_execute_single_word_hidden_state_batch_average_cli
from .study_single_word_top_100_neurons import register_cli as register_single_word_top_100_neurons_cli
from .study_single_word_top_100_neurons import try_execute_cli as try_execute_single_word_top_100_neurons_cli
from .study_token_diff import register_cli as register_token_diff_cli
from .study_token_diff import try_execute_cli as try_execute_token_diff_cli

BoolFlagParser = Callable[[str], bool]

REGISTER_STUDY_CLI_FUNCS = [
    register_single_word_hidden_state_cli,
    register_single_word_hidden_state_batch_average_cli,
    register_sentence_next_word_cli,
    register_token_diff_cli,
    register_one_on_one_attention_cli,
    register_chat_attention_word_replacement_cli,
    register_qk_params_cli,
    register_single_word_top_100_neurons_cli,
    register_layer_neuron_logits_table_cli,
    register_layer_ffn_neuron_logits_table_cli,
    register_layer_neurons_cli,
    register_layer_shortcut_cli,
    register_attribute_group_neurons_cli,
]

TRY_EXECUTE_STUDY_CLI_FUNCS = [
    try_execute_single_word_hidden_state_cli,
    try_execute_single_word_hidden_state_batch_average_cli,
    try_execute_sentence_next_word_cli,
    try_execute_token_diff_cli,
    try_execute_one_on_one_attention_cli,
    try_execute_chat_attention_word_replacement_cli,
    try_execute_qk_params_cli,
    try_execute_single_word_top_100_neurons_cli,
    try_execute_layer_neuron_logits_table_cli,
    try_execute_layer_ffn_neuron_logits_table_cli,
    try_execute_layer_neurons_cli,
    try_execute_layer_shortcut_cli,
    try_execute_attribute_group_neurons_cli,
]


def add_study_subparsers(subparsers: argparse._SubParsersAction, bool_parser: BoolFlagParser) -> None:
    for register in REGISTER_STUDY_CLI_FUNCS:
        register(subparsers, bool_parser)


def try_execute_study_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    for execute in TRY_EXECUTE_STUDY_CLI_FUNCS:
        payload = execute(args, config)
        if payload is not None:
            return payload
    return None
