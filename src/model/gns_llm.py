import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GNSLLM(torch.nn.Module):
    """
    Graph Neural Summarizer (GNS) integrated with a frozen LLM.
    
    GNS generates multiple query-aware prompt vectors from graph-structured
    data to mitigate the information bottleneck in graph-level prompting.
    
    Reference: Kim & Kim, "Addressing information bottlenecks in graph 
    augmented large language models via graph neural summarization",
    Information Fusion, 2026.
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '48GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=nf4_config,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        # Graph Neural Summarizer (GNS) encoder
        # Internally implements: GNNquery -> GNNnode -> Clustering -> GNNpool
        self.graph_encoder = load_gnn_model['gns'](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
            num_graph_token=args.num_graph_token,
            edge_feature=args.edge_feature,
            gnn=args.gnn_model_name
        ).to(self.model.device)

        # Projection layer: maps GNN output to LLM embedding space
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, self.model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

        # Pre-computed query embeddings from Sentence Transformer
        q_embs_path = f'dataset/{args.dataset}'
        self.q_embs = torch.tensor(
            torch.load(f'{q_embs_path}/q_embs.pt'), requires_grad=False
        )

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples, text_embedds=None, query_aware=True, pooling='graph_token'):
        """
        Encode graph using GNS to produce k query-aware prompt vectors.
        
        Args:
            samples: dict containing 'graph' key
            text_embedds: pre-computed query embeddings for query-aware encoding
            query_aware: whether to use query-aware mechanism (GNNquery)
            pooling: pooling strategy ('graph_token' for GNS multi-vector pooling)
        
        Returns:
            g_embeds: [batch_size, num_graph_token, hidden_dim]
        """
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        g_embeds, _ = self.graph_encoder(graphs, text_embedds, query_aware, pooling)
        return g_embeds

    def forward(self, samples, args):
        with self.maybe_autocast():
            # Encode description, questions and labels
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
            labels = self.tokenizer(samples["label"], add_special_tokens=False)

            # Encode special tokens
            eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
            eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
            bos_embeds = self.word_embedding(
                self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
            )
            pad_embeds = self.word_embedding(
                torch.tensor(self.tokenizer.pad_token_id)
            ).unsqueeze(0)

            # Encode graphs via GNS -> k prompt vectors
            graph_embeds = self.encode_graphs(
                samples,
                text_embedds=self.q_embs[samples['id']],
                query_aware=args.query_aware,
                pooling=args.pooling
            )
            graph_embeds = self.projector(graph_embeds)

            batch_size = len(samples['id'])
            batch_inputs_embeds = []
            batch_attention_mask = []
            batch_label_input_ids = []

            for i in range(batch_size):
                # Construct input: [BOS] [graph_tokens] [desc] [question] [EOS_USER] [label] [EOS]
                label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
                input_ids = (
                    descriptions.input_ids[i][:self.max_txt_len]
                    + questions.input_ids[i]
                    + eos_user_tokens.input_ids
                    + label_input_ids
                )
                inputs_embeds = self.word_embedding(
                    torch.tensor(input_ids).to(self.model.device)
                )

                # Concatenate: [BOS] + [GNS prompt tokens] + [text embeddings]
                inputs_embeds = torch.cat(
                    [bos_embeds, graph_embeds[i], inputs_embeds], dim=0
                )

                batch_inputs_embeds.append(inputs_embeds)
                batch_attention_mask.append([1] * inputs_embeds.shape[0])
                label_input_ids = (
                    [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids))
                    + label_input_ids
                )
                batch_label_input_ids.append(label_input_ids)

            # Pad inputs_embeds to max length in batch
            max_length = max([x.shape[0] for x in batch_inputs_embeds])
            for i in range(batch_size):
                pad_length = max_length - batch_inputs_embeds[i].shape[0]
                batch_inputs_embeds[i] = torch.cat(
                    [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
                )
                batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
                batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

            inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
            attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
            label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples, args):
        # Encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # Encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id)
        ).unsqueeze(0)

        # Encode graphs via GNS
        graph_embeds = self.encode_graphs(
            samples,
            text_embedds=self.q_embs[samples['id']],
            query_aware=args.query_aware,
            pooling=args.pooling
        )
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            # Construct input: [BOS] [graph_tokens] [desc] [question] [EOS_USER]
            input_ids = (
                descriptions.input_ids[i][:self.max_txt_len]
                + questions.input_ids[i]
                + eos_user_tokens.input_ids
            )
            inputs_embeds = self.word_embedding(
                torch.tensor(input_ids).to(self.model.device)
            )

            inputs_embeds = torch.cat(
                [bos_embeds, graph_embeds[i], inputs_embeds], dim=0
            )
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # Pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat(
                [pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]]
            )
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'id': samples['id'],
            'pred': pred,
            'label': samples['label'],
            'question': samples['question'],
            'desc': samples['desc'],
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
