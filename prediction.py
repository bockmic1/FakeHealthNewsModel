import torch
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import torch.nn as nn

class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert_content = BertModel(config)
        self.bert_FTR = BertModel(config)
        self.aggregator = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs_content = self.bert_content(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        outputs_FTR = self.bert_FTR(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        
        pooled_output_content = outputs_content[1]
        pooled_output_FTR = outputs_FTR[1]
        
        combined_output = self.aggregator(pooled_output_content + pooled_output_FTR)
        mlp_output = self.mlp(combined_output)
        
        logits = self.classifier(mlp_output)
        return logits

# Überprüfen, ob die Modellgewichte geladen sind
model_directory = './param_model/ARG_en-arg/1/'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = CustomBertForSequenceClassification.from_pretrained(model_directory)

for name, param in model.named_parameters():
    print(name, param.data)

# Eingabetext
input_text = "Human hair is cancerous"
inputs = tokenizer(input_text, return_tensors='pt')
model.eval()
with torch.no_grad():
    logits = model(**inputs)
probabilities = torch.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

if predicted_class == 0:
    print("The statement is classified as REAL.")
else:
    print("The statement is classified as FAKE.")
