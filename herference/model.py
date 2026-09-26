#  Copyright (c) 2022-2023 CLARIN-PL, Wroclaw University of Science and Technology
#  All rights reserved.
#
#  This file is free software: you may copy, redistribute and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This file is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#  This file incorporates work covered by the following copyright and
#  permission notice:
#    Copyright (c) 2021 (https://github.com/yuvalkirstain/s2e-coref/blob/main/modeling.py)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import logging
import math

import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, AutoModel
from transformers.activations import ACT2FN

from herference.utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters
from herference.utils_torch import split_tokenized, split_with_overlap
from herference.utils_torch import mask_tensor


logger = logging.getLogger(__name__)


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class S2E(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.max_span_length = params.max_span_length
        self.top_lambda = params.top_lambda
        self.ffnn_size = params.ffnn_size
        self.do_mlps = self.ffnn_size > 0
        self.ffnn_size = self.ffnn_size if self.do_mlps else config.hidden_size
        self.normalise_loss = params.normalise_loss
        self.max_model_input_length = params.max_model_input_length
        self.params = params
        logger.info(f"BERT _name_or_path: {config._name_or_path}")
        self.bert = AutoModel.from_config(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, params.dropout_prob) \
            if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, params.dropout_prob) \
            if self.do_mlps else None
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, params.dropout_prob) \
            if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, params.dropout_prob) \
            if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.antecedent_s2s_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2s_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.init_weights()

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * (
                    (seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        topk_mention_start_ids = torch.div(sorted_topk_1d_indices, seq_length,
                                           rounding_mode='floor')  # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
        topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(
            -2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(
                zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)
        ):
            gold_clusters = extract_clusters(gold_clusters, self.params.null_id_for_coref)
            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = mask_tensor(coref_logits, cluster_labels_after_pruning)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = - gold_log_probs
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size
        [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = \
            top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
        # [batch_size, max_k, max_k]
        return coref_logits

    # def forward(self, input_ids, attention_mask=None, gold_clusters=None, return_all_outputs=False):
    #     # logger.debug(f"model forward:\n input_ids: {input_ids.type()} {input_ids.shape} attention_mask:
    #     # {attention_mask.type()} {attention_mask.shape} gold_")
    #     n_tokens = input_ids.shape[-1]
    #
    #     if n_tokens <= self.max_model_input_length:
    #         outputs = self.bert(
    #             input_ids,
    #             attention_mask=attention_mask
    #         )
    #         sequence_output = outputs.last_hidden_state
    #     else:
    #         batch_outputs = []
    #
    #         for b in range(input_ids.shape[0]):
    #             input_ids_one = input_ids[b]
    #             attention_mask_one = attention_mask[b] if attention_mask is not None else None
    #
    #             chunk_list, spans = split_with_overlap(
    #                 input_ids_one,
    #                 chunk_size=400,
    #                 overlap=200,
    #                 min_chunk_size=50
    #             )
    #
    #             sequence_outputs = []
    #
    #             for i, chunk_ids in enumerate(chunk_list):
    #                 start, end = spans[i]
    #
    #                 if attention_mask_one is not None:
    #                     chunk_attn = attention_mask_one[start:end]
    #                 else:
    #                     chunk_attn = torch.ones_like(chunk_ids, dtype=torch.long)
    #
    #                 chunk_output = self.bert(
    #                     chunk_ids.unsqueeze(0),
    #                     attention_mask=chunk_attn.unsqueeze(0)
    #                 ).last_hidden_state.squeeze(0)  # [chunk_len, hidden_size]
    #
    #                 sequence_outputs.append(chunk_output)
    #
    #             sequence_output_one = torch.cat(sequence_outputs, dim=0)
    #             batch_outputs.append(sequence_output_one)
    #
    #         sequence_output = torch.stack(batch_outputs)  # [batch_size, total_len, hidden_size]
    #
    #     # Compute representations
    #     start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
    #     end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output
    #
    #     start_coref_reps = self.start_coref_mlp(sequence_output) if self.do_mlps else sequence_output
    #     end_coref_reps = self.end_coref_mlp(sequence_output) if self.do_mlps else sequence_output
    #
    #     # mention scores
    #     mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)
    #
    #     # prune mentions
    #     mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = \
    #         self._prune_topk_mentions(mention_logits, attention_mask)
    #
    #     batch_size, _, dim = start_coref_reps.size()
    #     max_k = mention_start_ids.size(-1)
    #     size = (batch_size, max_k, dim)
    #
    #     # Antecedent scores
    #     # gather reps
    #     topk_start_coref_reps = torch.gather(start_coref_reps, dim=1,
    #                                          index=mention_start_ids.unsqueeze(-1).expand(size))
    #     topk_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=mention_end_ids.unsqueeze(-1).expand(size))
    #     coref_logits = self._calc_coref_logits(topk_start_coref_reps, topk_end_coref_reps)
    #
    #     final_logits = topk_mention_logits + coref_logits
    #     final_logits = self._mask_antecedent_logits(final_logits, span_mask)
    #     # adding zero logits for null span
    #     final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1)
    #     # [batch_size, max_k, max_k + 1]
    #
    #     if return_all_outputs:
    #         outputs = (mention_start_ids, mention_end_ids, final_logits, mention_logits)
    #     else:
    #         outputs = tuple()
    #
    #     if gold_clusters is not None:
    #         losses = {}
    #         labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids,
    #                                                                       gold_clusters)
    #         loss = self._get_marginal_log_likelihood_loss(final_logits, labels_after_pruning, span_mask)
    #         losses.update({"loss": loss})
    #         outputs = (loss,) + outputs + (losses,)
    #
    #     return outputs

    def test_sliding_window_mapping(self, input_ids, attention_mask=None):
        """
        Testuje:
            globalny indeks -> lokalny indeks w oknie -> scalanie

        Sprawdza:
        1. czy każde okno ma poprawny zakres globalny,
        2. czy lokalny indeks odpowiada właściwemu tokenowi globalnemu,
        3. czy każde okno pokrywa właściwy zakres,
        4. czy po scaleniu każdy token globalny ma poprawną liczbę reprezentacji,
        5. czy scalanie overlapów wykonuje średnią reprezentacji.

        Test działa na jednym dokumencie.
        """

        if input_ids.dim() == 2:
            input_ids_one = input_ids[0]
            attention_mask_one = (
                attention_mask[0]
                if attention_mask is not None
                else None
            )
        else:
            input_ids_one = input_ids
            attention_mask_one = attention_mask

        seq_len = input_ids_one.size(0)

        chunk_size = 400
        overlap = 200
        min_chunk_size = 50

        print("\n" + "=" * 60)
        print("TEST GLOBALNY INDEKS -> LOKALNY INDEKS -> SCALANIE")
        print("=" * 60)

        print(f"Liczba tokenów: {seq_len}")
        print(f"chunk_size:     {chunk_size}")
        print(f"overlap:        {overlap}")

        # ---------------------------------------------------------
        # 1. Utworzenie okien
        # ---------------------------------------------------------

        chunk_list, spans = split_with_overlap(
            input_ids_one,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size
        )

        print(f"Liczba okien: {len(chunk_list)}")

        # ---------------------------------------------------------
        # 2. Sprawdzenie mapowania global -> lokalny
        # ---------------------------------------------------------

        coverage = [0] * seq_len

        mapping_errors = 0
        coverage_errors = 0

        for window_idx, (chunk_ids, (start, end)) in enumerate(
                zip(chunk_list, spans)
        ):

            chunk_len = chunk_ids.size(0)

            # Czy długość okna zgadza się z zakresem?
            if chunk_len != end - start:
                print(
                    f"\nBŁĄD DŁUGOŚCI OKNA {window_idx}:"
                )
                print(f"  start: {start}")
                print(f"  end:   {end}")
                print(f"  zakres: {end - start}")
                print(f"  chunk:  {chunk_len}")

                mapping_errors += 1

            # Sprawdź każdy lokalny indeks
            for local_idx in range(chunk_len):

                global_idx = start + local_idx

                # Globalny indeks musi być poprawny
                if global_idx < 0 or global_idx >= seq_len:
                    print(
                        f"\nBŁĄD GLOBALNEGO INDEKSU:"
                    )
                    print(f"  window: {window_idx}")
                    print(f"  local:  {local_idx}")
                    print(f"  global: {global_idx}")

                    mapping_errors += 1
                    continue

                # Czy token w oknie jest dokładnie tym samym
                # tokenem co token globalny?
                if chunk_ids[local_idx].item() != input_ids_one[global_idx].item():
                    print(
                        f"\nBŁĄD MAPOWANIA TOKENU:"
                    )
                    print(f"  window: {window_idx}")
                    print(f"  local:  {local_idx}")
                    print(f"  global: {global_idx}")
                    print(
                        f"  chunk token:  "
                        f"{chunk_ids[local_idx].item()}"
                    )
                    print(
                        f"  global token: "
                        f"{input_ids_one[global_idx].item()}"
                    )

                    mapping_errors += 1

            # Pokrycie
            for global_idx in range(start, end):
                coverage[global_idx] += 1

        # ---------------------------------------------------------
        # 3. Sprawdzenie pokrycia
        # ---------------------------------------------------------

        uncovered = [
            i for i, count in enumerate(coverage)
            if count == 0
        ]

        overcovered = [
            i for i, count in enumerate(coverage)
            if count > 2
        ]

        min_coverage = min(coverage)
        max_coverage = max(coverage)

        print("\nPOKRYCIE:")
        print(f"  minimalne: {min_coverage}")
        print(f"  maksymalne: {max_coverage}")
        print(f"  niepokryte: {len(uncovered)}")
        print(f"  >2 pokrycia: {len(overcovered)}")

        if uncovered:
            print(
                f"\nPierwsze niepokryte tokeny: "
                f"{uncovered[:20]}"
            )

            coverage_errors += len(uncovered)

        if overcovered:
            print(
                f"\nPierwsze tokeny z >2 pokryciami: "
                f"{overcovered[:20]}"
            )

            coverage_errors += len(overcovered)

        # ---------------------------------------------------------
        # 4. Test scalania reprezentacji
        # ---------------------------------------------------------

        print("\nTEST SCALANIA REPREZENTACJI...")

        hidden_size = 1

        representations = []

        for window_idx, (chunk_ids, (start, end)) in enumerate(
                zip(chunk_list, spans)
        ):
            chunk_len = end - start

            # Reprezentacja zawiera globalny indeks tokenu.
            chunk_repr = torch.arange(
                start,
                end,
                dtype=torch.float32,
                device=input_ids_one.device
            ).unsqueeze(-1)

            representations.append(chunk_repr)

        # ---------------------------------------------------------
        # Ręczne scalanie
        # ---------------------------------------------------------

        merged = torch.zeros(
            (seq_len, hidden_size),
            dtype=torch.float32,
            device=input_ids_one.device
        )

        counts = torch.zeros(
            seq_len,
            dtype=torch.float32,
            device=input_ids_one.device
        )

        for (chunk_repr, (start, end)) in zip(
                representations,
                spans
        ):
            merged[start:end] += chunk_repr
            counts[start:end] += 1

        merged = merged / counts.unsqueeze(-1)

        # ---------------------------------------------------------
        # 5. Sprawdzenie wyniku scalania
        # ---------------------------------------------------------

        expected = torch.arange(
            seq_len,
            dtype=torch.float32,
            device=input_ids_one.device
        ).unsqueeze(-1)

        merge_difference = torch.abs(
            merged - expected
        )

        max_difference = merge_difference.max().item()

        print(
            f"Maksymalna różnica po scaleniu: "
            f"{max_difference}"
        )

        merge_errors = 0

        if max_difference != 0:

            merge_errors = 1

            bad_positions = torch.nonzero(
                merge_difference.squeeze(-1) != 0
            ).flatten()

            print(
                "\nBŁĄD SCALANIA!"
            )

            print(
                "Pierwsze błędne pozycje:"
            )

            for idx in bad_positions[:20]:
                idx = idx.item()

                print(
                    f"  global={idx} "
                    f"oczekiwane={expected[idx].item()} "
                    f"otrzymane={merged[idx].item()}"
                )

        # ---------------------------------------------------------
        # 6. Szczegółowa kontrola kilku tokenów z overlapu
        # ---------------------------------------------------------

        print("\nPRZYKŁADOWE TOKENY Z OVERLAPU:")

        examples = []

        for window_idx in range(len(spans) - 1):

            start1, end1 = spans[window_idx]
            start2, end2 = spans[window_idx + 1]

            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)

            if overlap_start < overlap_end:
                examples.append(
                    (
                        window_idx,
                        window_idx + 1,
                        overlap_start,
                        overlap_end
                    )
                )

            if len(examples) >= 3:
                break

        for (
                w1,
                w2,
                overlap_start,
                overlap_end
        ) in examples:

            print(
                f"\nwindow {w1} <-> window {w2}"
            )

            print(
                f"overlap globalny: "
                f"{overlap_start}-{overlap_end - 1}"
            )

            for global_idx in range(
                    overlap_start,
                    min(overlap_start + 3, overlap_end)
            ):
                local1 = global_idx - spans[w1][0]
                local2 = global_idx - spans[w2][0]

                print(
                    f"  global={global_idx}"
                    f" | window {w1}: local={local1}"
                    f" | window {w2}: local={local2}"
                    f" | merged={merged[global_idx].item()}"
                )

        # ---------------------------------------------------------
        # 7. Wynik końcowy
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("PODSUMOWANIE TESTU SLIDING WINDOW")
        print("=" * 60)

        print(f"Błędy mapowania:       {mapping_errors}")
        print(f"Błędy pokrycia:        {coverage_errors}")
        print(f"Błędy scalania:        {merge_errors}")

        if (
                mapping_errors == 0
                and coverage_errors == 0
                and merge_errors == 0
        ):

            print(
                "\n✓ TEST SLIDING WINDOW: PASSED"
            )

            print(
                "✓ globalny -> lokalny indeks jest poprawny"
            )

            print(
                "✓ każde okno zawiera właściwe tokeny"
            )

            print(
                "✓ każdy token jest pokryty"
            )

            print(
                "✓ overlap ma maksymalnie 2 reprezentacje"
            )

            print(
                "✓ scalanie średnią jest poprawne"
            )

        else:

            print(
                "\n✗ TEST SLIDING WINDOW: FAILED"
            )

            raise RuntimeError(
                "Test sliding window nie przeszedł."
            )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            gold_clusters=None,
            return_all_outputs=False,
            debug_coref = False
    ):
        batch_size = input_ids.size(0)
        n_tokens = input_ids.size(-1)

        # ============================================================
        # FUNKCJA POMOCNICZA:
        # GOLD CLUSTERS GLOBALNE -> LOKALNE DLA OKNA
        # ============================================================

        def convert_gold_clusters_to_window(
                gold_clusters_one,
                window_start,
                window_end
        ):
            """
            gold_clusters_one:
                [num_clusters, max_mentions, 2]

            window:
                [window_start, window_end)

            Zwraca klastry w lokalnych indeksach okna.

            Zachowujemy tylko wzmianki znajdujące się CAŁKOWICIE
            w danym oknie.

            Przykład:

            global:
                [500, 510]

            okno:
                [400, 800]

            lokalnie:
                [100, 110]
            """

            if gold_clusters_one is None:
                return None

            local_clusters = []

            gold_cpu = gold_clusters_one.detach().cpu()

            for cluster in gold_cpu:

                local_cluster = []

                for mention in cluster:

                    global_start = int(mention[0].item())
                    global_end = int(mention[1].item())

                    # ------------------------------------------------
                    # Padding
                    # ------------------------------------------------

                    if global_start == 0 and global_end == 0:
                        continue

                    # ------------------------------------------------
                    # Niepoprawna wzmianka
                    # ------------------------------------------------

                    if global_start < 0 or global_end < global_start:
                        continue

                    # ------------------------------------------------
                    # Wzmianka musi być CAŁA w oknie
                    # ------------------------------------------------

                    if global_start < window_start:
                        continue

                    if global_end >= window_end:
                        continue

                    # ------------------------------------------------
                    # Globalny -> lokalny
                    # ------------------------------------------------

                    local_start = global_start - window_start
                    local_end = global_end - window_start

                    local_cluster.append(
                        [local_start, local_end]
                    )

                if local_cluster:
                    local_clusters.append(local_cluster)

            # --------------------------------------------------------
            # Tworzymy tensor o formacie:
            #
            # [num_clusters, max_mentions, 2]
            # --------------------------------------------------------

            if not local_clusters:
                return torch.zeros(
                    (1, 1, 2),
                    dtype=torch.long,
                    device=self.device
                )

            max_mentions = max(
                len(cluster)
                for cluster in local_clusters
            )

            local_gold = torch.zeros(
                (
                    len(local_clusters),
                    max_mentions,
                    2
                ),
                dtype=torch.long,
                device=self.device
            )

            for c_idx, cluster in enumerate(local_clusters):

                for m_idx, (start, end) in enumerate(cluster):
                    local_gold[c_idx, m_idx, 0] = start
                    local_gold[c_idx, m_idx, 1] = end

            return local_gold

        # ============================================================
        # FUNKCJA PRZETWARZAJĄCA JEDNO OKNO
        # ============================================================

        def process_window(
                chunk_ids,
                chunk_attention,
                window_start,
                window_end,
                gold_clusters_one=None
        ):
            """
            Przetwarza jedno okno.

            Zwraca:

                loss
                mention_start_ids
                mention_end_ids
                final_logits
                mention_logits
                span_mask
            """

            # --------------------------------------------------------
            # BERT
            # --------------------------------------------------------

            chunk_output = self.bert(
                chunk_ids.unsqueeze(0),
                attention_mask=chunk_attention.unsqueeze(0)
            )

            sequence_output = chunk_output.last_hidden_state
            # [1, window_size, hidden_size]

            window_size = sequence_output.size(1)

            # --------------------------------------------------------
            # MENTION REPRESENTATIONS
            # --------------------------------------------------------

            start_mention_reps = (
                self.start_mention_mlp(sequence_output)
                if self.do_mlps
                else sequence_output
            )

            end_mention_reps = (
                self.end_mention_mlp(sequence_output)
                if self.do_mlps
                else sequence_output
            )

            start_coref_reps = (
                self.start_coref_mlp(sequence_output)
                if self.do_mlps
                else sequence_output
            )

            end_coref_reps = (
                self.end_coref_mlp(sequence_output)
                if self.do_mlps
                else sequence_output
            )

            mention_logits = self._calc_mention_logits(
                start_mention_reps,
                end_mention_reps
            )

            # --------------------------------------------------------
            # PRUNING
            # --------------------------------------------------------

            (
                mention_start_ids,
                mention_end_ids,
                span_mask,
                topk_mention_logits
            ) = self._prune_topk_mentions(
                mention_logits,
                chunk_attention.unsqueeze(0)
            )

            # --------------------------------------------------------
            # COREFERENCE REPRESENTATIONS
            # --------------------------------------------------------

            max_k = mention_start_ids.size(-1)

            _, _, dim = start_coref_reps.size()

            gather_size = (
                1,
                max_k,
                dim
            )

            topk_start_coref_reps = torch.gather(
                start_coref_reps,
                dim=1,
                index=mention_start_ids
                .unsqueeze(-1)
                .expand(gather_size)
            )

            topk_end_coref_reps = torch.gather(
                end_coref_reps,
                dim=1,
                index=mention_end_ids
                .unsqueeze(-1)
                .expand(gather_size)
            )

            # --------------------------------------------------------
            # COREFERENCE LOGITS
            # --------------------------------------------------------

            coref_logits = self._calc_coref_logits(
                topk_start_coref_reps,
                topk_end_coref_reps
            )

            # --------------------------------------------------------
            # FINAL LOGITS
            # --------------------------------------------------------

            final_logits = (
                    topk_mention_logits +
                    coref_logits
            )

            final_logits = self._mask_antecedent_logits(
                final_logits,
                span_mask
            )

            # --------------------------------------------------------
            # NULL ANTECEDENT
            # --------------------------------------------------------

            null_logits = torch.zeros(
                (
                    1,
                    max_k,
                    1
                ),
                device=self.device,
                dtype=final_logits.dtype
            )

            final_logits = torch.cat(
                (
                    final_logits,
                    null_logits
                ),
                dim=-1
            )

            # --------------------------------------------------------
            # LOSS
            # --------------------------------------------------------

            loss = None

            if gold_clusters_one is not None:
                local_gold_clusters = (
                    convert_gold_clusters_to_window(
                        gold_clusters_one,
                        window_start,
                        window_end
                    )
                )

                labels_after_pruning = (
                    self._get_cluster_labels_after_pruning(
                        mention_start_ids,
                        mention_end_ids,
                        local_gold_clusters.unsqueeze(0)
                    )
                )

                loss = self._get_marginal_log_likelihood_loss(
                    final_logits,
                    labels_after_pruning,
                    span_mask
                )

            return (
                loss,
                mention_start_ids,
                mention_end_ids,
                final_logits,
                mention_logits,
                span_mask
            )

        # ============================================================
        # DEBUG
        # ============================================================

        if debug_coref:
            print("\n" + "=" * 70)
            print("S2E FORWARD")
            print("=" * 70)

            print(f"Batch size:             {batch_size}")
            print(f"Liczba tokenów:         {n_tokens}")
            print(f"Max BERT input:         {self.max_model_input_length}")

        # ============================================================
        # 1. DOKUMENT KRÓTKI
        # ============================================================

        if n_tokens <= self.max_model_input_length:

            total_loss = None

            outputs_all = []

            for b in range(batch_size):

                input_ids_one = input_ids[b]

                attention_mask_one = (
                    attention_mask[b]
                    if attention_mask is not None
                    else torch.ones_like(
                        input_ids_one,
                        dtype=torch.long
                    )
                )

                gold_one = None

                if gold_clusters is not None:
                    gold_one = gold_clusters[b]

                result = process_window(
                    input_ids_one,
                    attention_mask_one,
                    0,
                    n_tokens,
                    gold_one
                )

                (
                    loss,
                    mention_start_ids,
                    mention_end_ids,
                    final_logits,
                    mention_logits,
                    span_mask
                ) = result

                if loss is not None:

                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss = total_loss + loss

                outputs_all.append(result)

            if total_loss is not None:
                total_loss = total_loss / batch_size

            if return_all_outputs:

                result = outputs_all[0]

                outputs = (
                    result[1],  # mention_start_ids
                    result[2],  # mention_end_ids
                    result[3],  # final_logits
                    result[4],  # mention_logits
                    result[5]  # span_mask
                )

            else:
                outputs = tuple()

            if total_loss is not None:
                outputs = (
                              total_loss,
                          ) + outputs + (
                              {
                                  "loss": total_loss
                              },
                          )

            return outputs

        # ============================================================
        # 2. DŁUGI DOKUMENT
        # ============================================================

        batch_losses = []

        batch_outputs = []

        for b in range(batch_size):

            input_ids_one = input_ids[b]

            attention_mask_one = (
                attention_mask[b]
                if attention_mask is not None
                else torch.ones_like(
                    input_ids_one,
                    dtype=torch.long
                )
            )

            seq_len = input_ids_one.size(0)

            # --------------------------------------------------------
            # SLIDING WINDOW
            # --------------------------------------------------------

            chunk_list, spans = split_with_overlap(
                input_ids_one,
                chunk_size=400,
                overlap=200,
                min_chunk_size=50
            )

            if debug_coref:

                print(
                    f"SLIDING WINDOW: chunk=400, overlap=200"
                )

                print(
                    f"LICZBA OKIEN: {len(chunk_list)}"
                )

            document_losses = []

            document_outputs = []

            # --------------------------------------------------------
            # KAŻDE OKNO OSOBNO
            # --------------------------------------------------------

            for window_idx, (
                    chunk_ids,
                    (window_start, window_end)
            ) in enumerate(
                zip(chunk_list, spans)
            ):

                if debug_coref:
                    print(
                        f"\nWINDOW {window_idx + 1}/{len(chunk_list)}"
                        f" | global={window_start}:{window_end}"
                        f" | size={window_end - window_start}"
                    )

                if attention_mask_one is not None:

                    chunk_attention = (
                        attention_mask_one[
                        window_start:window_end
                        ]
                    )

                else:

                    chunk_attention = torch.ones(
                        window_end - window_start,
                        dtype=torch.long,
                        device=input_ids_one.device
                    )

                gold_one = None

                if gold_clusters is not None:
                    gold_one = gold_clusters[b]

                # ----------------------------------------------------
                # PRZETWARZANIE OKNA
                # ----------------------------------------------------

                result = process_window(
                    chunk_ids,
                    chunk_attention,
                    window_start,
                    window_end,
                    gold_one
                )

                (
                    loss,
                    mention_start_ids,
                    mention_end_ids,
                    final_logits,
                    mention_logits,
                    span_mask
                ) = result

                # ----------------------------------------------------
                # LOSS
                # ----------------------------------------------------

                if loss is not None:

                    if torch.isfinite(loss):

                        document_losses.append(loss)
                        if debug_coref:
                            print(
                                f"WINDOW LOSS: {loss.item():.6f}"
                            )

                    else:

                        print(
                            f"WARNING: NaN/Inf loss "
                            f"w window {window_idx}"
                        )

                # ----------------------------------------------------
                # OUTPUTS
                # ----------------------------------------------------

                if return_all_outputs:
                    document_outputs.append(
                        result
                    )

                # ----------------------------------------------------
                # DEBUG
                # ----------------------------------------------------

                if debug_coref:
                    print(
                        f"mention_logits: "
                        f"{mention_logits.shape}"
                    )

                    print(
                        f"pruned mentions: "
                        f"{span_mask.sum().item():.0f}"
                    )

            # ========================================================
            # LOSS DLA DOKUMENTU
            # ========================================================

            if document_losses:
                document_loss = torch.stack(
                    document_losses
                ).mean()

                batch_losses.append(
                    document_loss
                )

                print(
                    f"\nDOCUMENT LOSS: "
                    f"{document_loss.item():.6f}"
                )

            # ========================================================
            # OUTPUTS DOKUMENTU
            # ========================================================

            if return_all_outputs:
                batch_outputs.append(
                    document_outputs
                )

        # ============================================================
        # 3. GLOBALNY LOSS
        # ============================================================

        if batch_losses:

            loss = torch.stack(
                batch_losses
            ).mean()

            print(
                "\n" +
                "=" * 60
            )

            print(
                f"GLOBAL LOSS: {loss.item():.6f}"
            )

            print(
                "=" * 60
            )

        else:

            loss = None

        # ============================================================
        # 4. OUTPUT
        # ============================================================

        if return_all_outputs:

            # Dla długich dokumentów mamy wiele okien.
            # Zwracamy listę wyników z poszczególnych okien.
            outputs = batch_outputs

        else:

            outputs = tuple()

        if loss is not None:
            outputs = (
                          loss,
                      ) + (
                          outputs if isinstance(outputs, tuple)
                          else (outputs,)
                      ) + (
                          {
                              "loss": loss
                          },
                      )

        return outputs
