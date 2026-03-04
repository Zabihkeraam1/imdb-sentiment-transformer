import torch.nn as nn
from transformers import AutoModel


class SentimentModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(SentimentModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        class Output:
            pass

        result = Output()
        result.logits = logits
        result.loss = loss
        return result