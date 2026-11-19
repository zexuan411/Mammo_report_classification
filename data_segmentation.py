import pandas as pd
import re
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel

# Inline clean_text function to avoid torch import
_whitespace_re = re.compile(r"[ \t]+")
_newlines_re = re.compile(r"\n{2,}")

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # s = str(s).strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[?\\/]{1} ?", "", s)
    # s = re.sub(r"([.,:;!?])\1+", r"\1", s)
    # s = re.sub(r"[®]", "", s)
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n", s)
    return s


def split_report_by_sections(report):
    """
    Split report into three sections: Indicação clínica, Achados, Análise comparativa
    """
    sections = {
        'clinical_indication': '',
        'findings': '',
        'comparative_analysis': ''
    }
    
    report = clean_text(report)
    report_lower = report.lower()
    
    clinical_idx = -1
    clinical_variants = ['indicação clínica', 'indicação', 'ndicação clínica', 'indicaclica', 'indicaclca']  
    for variant in clinical_variants:
        idx = report_lower.find(variant)
        if idx != -1:
            clinical_idx = idx
            break
    
    # Find findings start index
    findings_idx = -1
    findings_variants = ['achados', 'descrição do exame', 'análise', 'chados']
    for variant in findings_variants:
        idx = report_lower.find(variant)
        if idx != -1:
            findings_idx = idx
            break
    
    # Find comparative analysis start index
    comparative_idx = report_lower.find('análise comparativa')
    
    # ============ Extract clinical indication ============
    if clinical_idx != -1:
        colon_idx = report.find(':', clinical_idx)
        if colon_idx != -1:
            # clinical indication starts after ":"
            if findings_idx != -1 and findings_idx > colon_idx:
                sections['clinical_indication'] = report[colon_idx+1:findings_idx].strip()
            elif comparative_idx != -1 and comparative_idx > colon_idx:
                sections['clinical_indication'] = report[colon_idx+1:comparative_idx].strip()
            else:
                sections['clinical_indication'] = report[colon_idx+1:].strip()
    
    # ============ Extract findings ============
    # Case 1: explicit "Achados:" label
    if findings_idx != -1:
        colon_idx = report.find(':', findings_idx)
        if colon_idx != -1:
            # findings start after ":"
            if comparative_idx != -1 and comparative_idx > colon_idx:
                sections['findings'] = report[colon_idx+1:comparative_idx].strip()
            else:
                sections['findings'] = report[colon_idx+1:].strip()
        else:
            # No ":", start right after the heading
            if comparative_idx != -1 and comparative_idx > findings_idx:
                sections['findings'] = report[findings_idx:comparative_idx].strip()
            else:
                sections['findings'] = report[findings_idx:].strip()
    # Case 2: no "Achados:" label; start after first "." of clinical indication
    elif clinical_idx != -1:
        colon_idx = report.find(':', clinical_idx)
        if colon_idx != -1:
            # Use text after the first "." of clinical indication as findings
            dot_idx = report.find('.', colon_idx)
            if dot_idx != -1:
                if comparative_idx != -1 and comparative_idx > dot_idx:
                    sections['findings'] = report[dot_idx+1:comparative_idx].strip()
                else:
                    sections['findings'] = report[dot_idx+1:].strip()
    
    # ============ Extract comparative analysis ============
    if comparative_idx != -1:
        colon_idx = report.find(':', comparative_idx)
        if colon_idx != -1:
            sections['comparative_analysis'] = report[colon_idx+1:].strip()
        else:
            sections['comparative_analysis'] = report[comparative_idx:].strip()
    
    # ============ Clean each section ============
    for key in sections:
        text = sections[key]
        # Remove residual section titles (e.g. "Indicação clínica:")
        text = re.sub(r'^[^:]*:\s*', '', text, count=1)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r"([.,:;!?])([.,:;!?])+", r"\1", text)
        sections[key] = text
    
    return sections


class DualHeadOutput(NamedTuple):
    """CE / hierarchical logits + optional ordinal (+ binary) heads, matching bertori.py.

    NamedTuple so DDP ``find_unused_parameters`` can recurse into the returned
    tensors. A plain class is invisible to ``_find_tensors``, which marks
    ``ordinal_classifier`` unused and then ready again on backward.
    """

    logits: torch.Tensor
    ordinal_logits: Optional[torch.Tensor] = None
    binary_logits: Optional[torch.Tensor] = None


class TokenLevelSegmentedBert(nn.Module):
    def __init__(self, model_name, num_labels,
                 hidden_size=768,
                 proj_size=384,
                 dropout_rate=0.1,
                 transformer_layers=1,
                 nhead=4,
                 dual_head=False,
                 no_segment_fusion=False):
        super(TokenLevelSegmentedBert, self).__init__()

        if num_labels < 2 and dual_head:
            raise ValueError("dual_head requires num_labels >= 2")

        self.num_labels = num_labels
        self.dual_head = bool(dual_head)
        self.no_segment_fusion = bool(no_segment_fusion)

        # --- BERT backbone ---
        self.bert = AutoModel.from_pretrained(model_name)

        # --- Projection ---
        self.proj = nn.Linear(hidden_size, proj_size)

        # --- Segment embeddings ---
        self.segment_embed = nn.Embedding(3, proj_size)

        # Dynamic segment gating
        self.segment_gate = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.segment_gate.weight)
        nn.init.zeros_(self.segment_gate.bias)

        # --- Global CLS ---
        self.global_cls = nn.Parameter(
            torch.empty(1, 1, proj_size).normal_(std=np.sqrt(1. / proj_size))
        )

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_size,
            nhead=nhead,
            dim_feedforward=proj_size * 2,
            dropout=dropout_rate,
            activation='relu'
        )
        self.cross_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # --- Attention pooling ---
        self.attn_pool = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.attn_pool.weight)
        nn.init.zeros_(self.attn_pool.bias)

        # --- Classifier (CE head); optional ordinal aux head like bertori DualHead ---
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(proj_size, num_labels)
        self.ordinal_classifier = (
            nn.Linear(proj_size, num_labels - 1) if self.dual_head else None
        )

    def _run_transformer_and_capture_last_attn(self, x):
        """
        Run TransformerEncoder while capturing self-attention weights from the last layer.
        x shape: (seq_len, batch, dim)
        Returns:
            x_out: (seq_len, batch, dim)
            last_attn_weights: (batch, heads, seq_len, seq_len) or None
        """
        if len(self.cross_transformer.layers) == 0:
            return x, None

        last_attn_weights = None
        for i, layer in enumerate(self.cross_transformer.layers):
            is_last = (i == len(self.cross_transformer.layers) - 1)
            if not is_last:
                x = layer(x)
                continue

            if layer.norm_first:
                x_norm = layer.norm1(x)
                sa_out, last_attn_weights = layer.self_attn(
                    x_norm,
                    x_norm,
                    x_norm,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                )
                x = x + layer.dropout1(sa_out)
                x = x + layer._ff_block(layer.norm2(x))
            else:
                sa_out, last_attn_weights = layer.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                )
                x = layer.norm1(x + layer.dropout1(sa_out))
                x = layer.norm2(x + layer._ff_block(x))

        if self.cross_transformer.norm is not None:
            x = self.cross_transformer.norm(x)

        return x, last_attn_weights

    def forward(self,
                clinical_input_ids, clinical_attention_mask,
                findings_input_ids, findings_attention_mask,
                comparative_input_ids, comparative_attention_mask,
                return_last_layer_attn=False):

        batch_size = clinical_input_ids.size(0)

        # ===== 1. BERT encoding =====
        clinical_tokens = self.bert(
            clinical_input_ids,
            attention_mask=clinical_attention_mask
        ).last_hidden_state

        findings_tokens = self.bert(
            findings_input_ids,
            attention_mask=findings_attention_mask
        ).last_hidden_state

        comparative_tokens = self.bert(
            comparative_input_ids,
            attention_mask=comparative_attention_mask
        ).last_hidden_state

        # ===== 2. Projection =====
        clinical_proj = self.proj(clinical_tokens)
        findings_proj = self.proj(findings_tokens)
        comparative_proj = self.proj(comparative_tokens)

        if self.no_segment_fusion:
            # --- Ablation: skip segment embed / gating / transformer / attn pool ---
            # Use CLS token ([0]) from each projected segment, then mean-pool
            final_repr = (clinical_proj[:, 0] + findings_proj[:, 0] + comparative_proj[:, 0]) / 3.0
            last_layer_attn = None
        else:
            # ===== 3. Segment embedding =====
            clinical_proj = clinical_proj + self.segment_embed(torch.zeros_like(clinical_input_ids))
            findings_proj = findings_proj + self.segment_embed(torch.ones_like(findings_input_ids))
            comparative_proj = comparative_proj + self.segment_embed(torch.ones_like(comparative_input_ids) * 2)

            # ===== 4. Dynamic segment weights =====
            clinical_vec = clinical_proj.mean(dim=1)
            findings_vec = findings_proj.mean(dim=1)
            comparative_vec = comparative_proj.mean(dim=1)

            segment_scores = torch.cat([
                self.segment_gate(clinical_vec),
                self.segment_gate(findings_vec),
                self.segment_gate(comparative_vec)
            ], dim=1)

            segment_weights = torch.softmax(segment_scores, dim=1)

            w1 = segment_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
            w2 = segment_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
            w3 = segment_weights[:, 2].unsqueeze(-1).unsqueeze(-1)

            clinical_proj = clinical_proj * w1
            findings_proj = findings_proj * w2
            comparative_proj = comparative_proj * w3

            # ===== 5. Concatenate tokens =====
            x = torch.cat([clinical_proj, findings_proj, comparative_proj], dim=1)

            # ===== 6. Add Global CLS =====
            cls_token = self.global_cls.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

            # ===== 7. Transformer =====
            x = x.permute(1, 0, 2)
            if return_last_layer_attn:
                x, last_layer_attn = self._run_transformer_and_capture_last_attn(x)
            else:
                x = self.cross_transformer(x)
                last_layer_attn = None

            # ===== 8. CLS + token =====
            cls_output = x[0]
            token_output = x[1:]

            # ===== 9. Attention pooling =====
            attn_scores = self.attn_pool(token_output)
            attn_weights = torch.softmax(attn_scores, dim=0)
            attn_pooled = (token_output * attn_weights).sum(dim=0)

            final_repr = cls_output + attn_pooled
        # dropped = self.dropout(final_repr)
        logits = self.classifier(final_repr)
        if self.dual_head:
            outputs = DualHeadOutput(
                logits=logits,
                ordinal_logits=self.ordinal_classifier(final_repr),
            )
        else:
            outputs = logits

        if return_last_layer_attn:
            return outputs, last_layer_attn
        return outputs

# seperete projection

class TokenLevelSegmentedBert1(nn.Module):
    def __init__(self, model_name, num_labels,
                 hidden_size=768,
                 proj_size=384,
                 dropout_rate=0.1,
                 transformer_layers=1,
                 nhead=4):
        super(TokenLevelSegmentedBert, self).__init__()

        # --- BERT backbone (Shared) ---
        self.bert = AutoModel.from_pretrained(model_name)

        # --- Separate Projection layers ---
        self.clinical_proj_layer = nn.Linear(hidden_size, proj_size)
        self.findings_proj_layer = nn.Linear(hidden_size, proj_size)
        self.comparative_proj_layer = nn.Linear(hidden_size, proj_size)

        # --- Segment embeddings ---
        self.segment_embed = nn.Embedding(3, proj_size)

        # Dynamic segment gating
        self.segment_gate = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.segment_gate.weight)
        nn.init.zeros_(self.segment_gate.bias)

        # --- Global CLS ---
        self.global_cls = nn.Parameter(
            torch.empty(1, 1, proj_size).normal_(std=np.sqrt(1. / proj_size))
        )

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_size,
            nhead=nhead,
            dim_feedforward=proj_size * 2,
            dropout=dropout_rate,
            activation='relu'
        )
        self.cross_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # --- Attention pooling ---
        self.attn_pool = nn.Linear(proj_size, 1)
        nn.init.xavier_uniform_(self.attn_pool.weight)
        nn.init.zeros_(self.attn_pool.bias)

        # --- Classifier ---
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(proj_size, num_labels)

    def _run_transformer_and_capture_last_attn(self, x):
        """
        Run TransformerEncoder while capturing self-attention weights from the last layer.
        x shape: (seq_len, batch, dim)
        Returns:
            x_out: (seq_len, batch, dim)
            last_attn_weights: (batch, heads, seq_len, seq_len) or None
        """
        if len(self.cross_transformer.layers) == 0:
            return x, None

        last_attn_weights = None
        for i, layer in enumerate(self.cross_transformer.layers):
            is_last = (i == len(self.cross_transformer.layers) - 1)
            if not is_last:
                x = layer(x)
                continue

            if layer.norm_first:
                x_norm = layer.norm1(x)
                sa_out, last_attn_weights = layer.self_attn(
                    x_norm,
                    x_norm,
                    x_norm,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                )
                x = x + layer.dropout1(sa_out)
                x = x + layer._ff_block(layer.norm2(x))
            else:
                sa_out, last_attn_weights = layer.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False,
                )
                x = layer.norm1(x + layer.dropout1(sa_out))
                x = layer.norm2(x + layer._ff_block(x))

        if self.cross_transformer.norm is not None:
            x = self.cross_transformer.norm(x)

        return x, last_attn_weights

    def forward(self,
                clinical_input_ids, clinical_attention_mask,
                findings_input_ids, findings_attention_mask,
                comparative_input_ids, comparative_attention_mask,
                return_last_layer_attn=False):

        batch_size = clinical_input_ids.size(0)

        # ===== 1. BERT encoding =====
        clinical_tokens = self.bert(
            clinical_input_ids,
            attention_mask=clinical_attention_mask
        ).last_hidden_state

        findings_tokens = self.bert(
            findings_input_ids,
            attention_mask=findings_attention_mask
        ).last_hidden_state

        comparative_tokens = self.bert(
            comparative_input_ids,
            attention_mask=comparative_attention_mask
        ).last_hidden_state

        # ===== 2. Projection (Separate for each segment) =====
        clinical_proj = self.clinical_proj_layer(clinical_tokens)
        findings_proj = self.findings_proj_layer(findings_tokens)
        comparative_proj = self.comparative_proj_layer(comparative_tokens)

        # ===== 3. Segment embedding =====
        clinical_proj = clinical_proj + self.segment_embed(torch.zeros_like(clinical_input_ids))
        findings_proj = findings_proj + self.segment_embed(torch.ones_like(findings_input_ids))
        comparative_proj = comparative_proj + self.segment_embed(torch.ones_like(comparative_input_ids) * 2)

        # ===== 4. Dynamic segment weights =====
        # (1) Get per-segment representation (mean pooling)
        clinical_vec = clinical_proj.mean(dim=1)      # (batch, proj_size)
        findings_vec = findings_proj.mean(dim=1)
        comparative_vec = comparative_proj.mean(dim=1)

        # (2) Compute weights
        segment_scores = torch.cat([
            self.segment_gate(clinical_vec),
            self.segment_gate(findings_vec),
            self.segment_gate(comparative_vec)
        ], dim=1)  # (batch, 3)

        segment_weights = torch.softmax(segment_scores, dim=1)  # (batch, 3)

        # (3) reshape for broadcasting
        w1 = segment_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        w2 = segment_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        w3 = segment_weights[:, 2].unsqueeze(-1).unsqueeze(-1)

        # (4) Weight tokens
        clinical_proj = clinical_proj * w1
        findings_proj = findings_proj * w2
        comparative_proj = comparative_proj * w3

        # ===== 5. Concatenate tokens =====
        x = torch.cat([clinical_proj, findings_proj, comparative_proj], dim=1)

        # ===== 6. Add Global CLS =====
        cls_token = self.global_cls.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # ===== 7. Transformer =====
        x = x.permute(1, 0, 2)
        if return_last_layer_attn:
            x, last_layer_attn = self._run_transformer_and_capture_last_attn(x)
        else:
            x = self.cross_transformer(x)
            last_layer_attn = None

        # ===== 8. CLS + token =====
        cls_output = x[0]
        token_output = x[1:]

        # ===== 9. Attention pooling =====
        attn_scores = self.attn_pool(token_output)
        attn_weights = torch.softmax(attn_scores, dim=0)
        attn_pooled = (token_output * attn_weights).sum(dim=0)

        # Fusion
        final_repr = cls_output + attn_pooled

        logits = self.classifier(self.dropout(final_repr))
        if return_last_layer_attn:
            return logits, last_layer_attn
        return logits


class SegmentedTextDataset(Dataset):
    """
    Dataset for segmented report text with labels
    Each sample contains three segments: clinical_indication, findings, comparative_analysis
    """
    def __init__(self, encodings_list, labels=None, carcinoma_flags=None):
        """
        Args:
            encodings_list: List of tuples, each containing encodings for [clinical_indication, findings, comparative_analysis]
            labels: Optional tensor of labels
        """
        self.encodings_list = encodings_list
        self.labels = labels
        self.carcinoma_flags = carcinoma_flags
    
    def __len__(self):
        return len(self.encodings_list)
    
    def __getitem__(self, idx):
        item = {}
        # Store three segments
        cli_enc, find_enc, comp_enc = self.encodings_list[idx]
        
        item['clinical_input_ids'] = cli_enc['input_ids']
        item['clinical_attention_mask'] = cli_enc['attention_mask']
        
        item['findings_input_ids'] = find_enc['input_ids']
        item['findings_attention_mask'] = find_enc['attention_mask']
        
        item['comparative_input_ids'] = comp_enc['input_ids']
        item['comparative_attention_mask'] = comp_enc['attention_mask']
        
        if self.labels is not None:
            item['labels'] = self.labels[idx]

        if self.carcinoma_flags is not None:
            item['carcinoma_flag'] = self.carcinoma_flags[idx]
        
        return item


def make_segmented_encodings(
    tokenizer,
    clinical_texts,
    findings_texts,
    comparative_texts,
    max_length,
    zero_empty_masks=False,
):
    """
    Tokenize pre-segmented reports (from train_split.csv)
    
    Args:
        tokenizer: HuggingFace tokenizer
        clinical_texts: List of clinical indication texts
        findings_texts: List of findings texts
        comparative_texts: List of comparative analysis texts
        max_length: Maximum token length for each segment
        zero_empty_masks: If True, set attention_mask to all-zeros for empty
            segment texts (so padded CLS/SEP from "" do not contribute).
    
    Returns:
        List of tuples (clinical_enc, findings_enc, comparative_enc)
    """
    encodings_list = []
    
    for cli_text, find_text, comp_text in zip(clinical_texts, findings_texts, comparative_texts):
        # Handle NaN values (empty comparative_analysis)
        cli_text = "" if pd.isna(cli_text) else str(cli_text)
        find_text = "" if pd.isna(find_text) else str(find_text)
        comp_text = "" if pd.isna(comp_text) else str(comp_text)
        
        # Tokenize each section
        clinical_enc = tokenizer(
            cli_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        findings_enc = tokenizer(
            find_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        comparative_enc = tokenizer(
            comp_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert tensors to CPU and remove batch dimension
        clinical_enc = {k: v.squeeze(0) for k, v in clinical_enc.items()}
        findings_enc = {k: v.squeeze(0) for k, v in findings_enc.items()}
        comparative_enc = {k: v.squeeze(0) for k, v in comparative_enc.items()}

        if zero_empty_masks:
            if not cli_text.strip():
                clinical_enc["attention_mask"] = torch.zeros_like(
                    clinical_enc["attention_mask"]
                )
            if not find_text.strip():
                findings_enc["attention_mask"] = torch.zeros_like(
                    findings_enc["attention_mask"]
                )
            if not comp_text.strip():
                comparative_enc["attention_mask"] = torch.zeros_like(
                    comparative_enc["attention_mask"]
                )
        
        encodings_list.append((clinical_enc, findings_enc, comparative_enc))
    
    return encodings_list
