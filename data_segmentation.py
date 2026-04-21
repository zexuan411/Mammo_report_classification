import pandas as pd
import re
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
    
    # 寻找 findings 的起始位置
    findings_idx = -1
    findings_variants = ['achados', 'descrição do exame', 'análise', 'chados']
    for variant in findings_variants:
        idx = report_lower.find(variant)
        if idx != -1:
            findings_idx = idx
            break
    
    # 寻找 comparative analysis 的起始位置
    comparative_idx = report_lower.find('análise comparativa')
    
    # ============ 提取 clinical indication ============
    if clinical_idx != -1:
        colon_idx = report.find(':', clinical_idx)
        if colon_idx != -1:
            # clinical indication 从 ":" 后开始
            if findings_idx != -1 and findings_idx > colon_idx:
                sections['clinical_indication'] = report[colon_idx+1:findings_idx].strip()
            elif comparative_idx != -1 and comparative_idx > colon_idx:
                sections['clinical_indication'] = report[colon_idx+1:comparative_idx].strip()
            else:
                sections['clinical_indication'] = report[colon_idx+1:].strip()
    
    # ============ 提取 findings ============
    # 情况1: 有明确的 "Achados:" 标签
    if findings_idx != -1:
        colon_idx = report.find(':', findings_idx)
        if colon_idx != -1:
            # findings 从 ":" 后开始
            if comparative_idx != -1 and comparative_idx > colon_idx:
                sections['findings'] = report[colon_idx+1:comparative_idx].strip()
            else:
                sections['findings'] = report[colon_idx+1:].strip()
        else:
            # 没有 ":", 直接从标题后开始
            if comparative_idx != -1 and comparative_idx > findings_idx:
                sections['findings'] = report[findings_idx:comparative_idx].strip()
            else:
                sections['findings'] = report[findings_idx:].strip()
    # 情况2: 没有明确的 "Achados:" 标签，从 clinical indication 的第一个"."之后开始
    elif clinical_idx != -1:
        colon_idx = report.find(':', clinical_idx)
        if colon_idx != -1:
            # 从 clinical indication 的第一个"."后开始作为 findings
            dot_idx = report.find('.', colon_idx)
            if dot_idx != -1:
                if comparative_idx != -1 and comparative_idx > dot_idx:
                    sections['findings'] = report[dot_idx+1:comparative_idx].strip()
                else:
                    sections['findings'] = report[dot_idx+1:].strip()
    
    # ============ 提取 comparative analysis ============
    if comparative_idx != -1:
        colon_idx = report.find(':', comparative_idx)
        if colon_idx != -1:
            sections['comparative_analysis'] = report[colon_idx+1:].strip()
        else:
            sections['comparative_analysis'] = report[comparative_idx:].strip()
    
    # ============ 清理每个部分 ============
    for key in sections:
        text = sections[key]
        # 移除任何可能的标题残留（例如 "Indicação clínica:"）
        text = re.sub(r'^[^:]*:\s*', '', text, count=1)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r"([.,:;!?])([.,:;!?])+", r"\1", text)
        sections[key] = text
    
    return sections


class TokenLevelSegmentedBert(nn.Module):
    def __init__(self, model_name, num_labels,
                 hidden_size=768,
                 proj_size=384,
                 dropout_rate=0.1,
                 transformer_layers=1,
                 nhead=4,
                 carcinoma_constraint_w_init=0.1):
        super(TokenLevelSegmentedBert, self).__init__()

        # --- BERT backbone ---
        self.bert = AutoModel.from_pretrained(model_name)

        # --- Projection ---
        self.proj = nn.Linear(hidden_size, proj_size)

        # --- Segment embeddings ---
        self.segment_embed = nn.Embedding(3, proj_size)

        #  动态段落权重（gating）
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

        # Learnable non-negative weight (via softplus) for optional carcinoma logit constraint loss.
        w_init = max(float(carcinoma_constraint_w_init), 1e-6)
        raw_init = np.log(np.expm1(w_init))

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
                return_constraint_weight=False,
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

        # ===== 3. Segment embedding =====
        clinical_proj = clinical_proj + self.segment_embed(torch.zeros_like(clinical_input_ids))
        findings_proj = findings_proj + self.segment_embed(torch.ones_like(findings_input_ids))
        comparative_proj = comparative_proj + self.segment_embed(torch.ones_like(comparative_input_ids) * 2)

        # ===== 4. 动态段落权重 =====
        # (1) 先得到每段 representation（mean pooling）
        clinical_vec = clinical_proj.mean(dim=1)      # (batch, proj_size)
        findings_vec = findings_proj.mean(dim=1)
        comparative_vec = comparative_proj.mean(dim=1)

        # (2) 计算权重
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

        # (4) 对 token 加权
        clinical_proj = clinical_proj * w1
        findings_proj = findings_proj * w2
        comparative_proj = comparative_proj * w3

        # ===== 5. 拼接 token =====
        x = torch.cat([clinical_proj, findings_proj, comparative_proj], dim=1)

        # ===== 6. 加 Global CLS =====
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

        # 融合
        final_repr = cls_output + attn_pooled

        logits = self.classifier(self.dropout(final_repr))
        if return_constraint_weight and return_last_layer_attn:
            return logits, self.get_carcinoma_constraint_w(), last_layer_attn
        if return_constraint_weight:
            return logits, self.get_carcinoma_constraint_w()
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


def make_segmented_encodings(tokenizer, clinical_texts, findings_texts, comparative_texts, max_length):
    """
    Tokenize pre-segmented reports (from train_split.csv)
    
    Args:
        tokenizer: HuggingFace tokenizer
        clinical_texts: List of clinical indication texts
        findings_texts: List of findings texts
        comparative_texts: List of comparative analysis texts
        max_length: Maximum token length for each segment
    
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
        
        encodings_list.append((clinical_enc, findings_enc, comparative_enc))
    
    return encodings_list

# data_path = "./train_new.csv"
# df = pd.read_csv(data_path)
# print(f"总报告数: {len(df)}\n")

# missing_clinical = 0
# missing_findings = 0
# missing_comparative = 0
# all_missing = 0
# partial_missing = 0
# debug_examples = []

# # 存储分段后的数据
# segmented_data = []

# for idx, row in df.iterrows():
#     report_id = row['ID']
#     report_raw = row['report']
#     target = row['target']
    
#     # 调用分段函数（内部会调用clean_text）
#     sections = split_report_by_sections(report_raw)
    
#     miss = []
#     if not sections['clinical_indication'] or sections['clinical_indication'].strip() == '':
#         missing_clinical += 1
#         # save report to file
#         open(f"missing_clinical_{report_id}.txt", "w").write(report_raw)
#         miss.append('clinical')
    
#     if not sections['findings'] or sections['findings'].strip() == '':
#         missing_findings += 1
#         miss.append('findings')
    
#     if not sections['comparative_analysis'] or sections['comparative_analysis'].strip() == '':
#         missing_comparative += 1
#         miss.append('comparative')
    
#     # 记录缺失情况
#     if len(miss) == 3:
#         all_missing += 1
#         if len(debug_examples) < 5:
#             cleaned = clean_text(report_raw)
#             debug_examples.append((report_id, report_raw[:100], cleaned[:100], sections))
#     elif len(miss) > 0:
#         partial_missing += 1
    
#     # 收集分段数据
#     segmented_data.append({
#         'ID': report_id,
#         'target': target,
#         'clinical_indication': sections['clinical_indication'],
#         'findings': sections['findings'],
#         'comparative_analysis': sections['comparative_analysis']
#     })

# print("="*80)
# print("分段结果统计")
# print("="*80)
# print(f"缺失 clinical_indication 的报告数: {missing_clinical}")
# print(f"缺失 findings 的报告数: {missing_findings}")
# print(f"缺失 comparative_analysis 的报告数: {missing_comparative}")

# # 保存分段后的数据到CSV
# segmented_df = pd.DataFrame(segmented_data)
# output_path = "./train_split.csv"
# segmented_df.to_csv(output_path, index=False)
# print(f"\n已保存分段数据到: {output_path}")
# print(f"数据形状: {segmented_df.shape}")
# print(f"\n前5行样本:")
# print(segmented_df.head())
