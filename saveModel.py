import torch
import os
from transformers import BertConfig, BertPreTrainedModel, BertModel
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

# Pfad zur .pkl-Datei
pkl_path = './param_model/ARG_en-arg/1/parameter_bert.pkl'

# Verzeichnis, in dem die Dateien gespeichert werden
model_directory = './param_model/ARG_en-arg/1/'

# Modellgewichte aus der .pkl-Datei laden
model_state_dict = torch.load(pkl_path, map_location=torch.device('cpu'))

# Konfigurationsinformationen (angepasst an Ihre Konfiguration)
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2  # Passen Sie dies an Ihre Klassifikationsaufgabe an

# Angepasstes Modell mit der Konfiguration initialisieren und die Gewichte laden
model = CustomBertForSequenceClassification(config)
model.load_state_dict(model_state_dict, strict=False)

# Verzeichnis erstellen, falls es nicht existiert
os.makedirs(model_directory, exist_ok=True)

# Modellgewichte manuell speichern
print("Saving model weights manually...")
torch.save(model.state_dict(), os.path.join(model_directory, 'pytorch_model.bin'))
print("Model weights saved manually.")

# Konfigurationsdatei als config.json speichern
print("Saving configuration...")
config.save_pretrained(model_directory)
print("Configuration saved successfully.")

# Überprüfen, ob die Datei pytorch_model.bin existiert und nicht leer ist
bin_path = os.path.join(model_directory, 'pytorch_model.bin')
if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
    print("Model file pytorch_model.bin exists and is not empty.")
else:
    print("Model file pytorch_model.bin is missing or empty.")
