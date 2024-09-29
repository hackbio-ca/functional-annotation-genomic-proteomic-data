import torch
import numpy as np
import os
import torch.nn as nn

project_folder = "/home/alex/760Gb_ssd/Bio_hawk_tuahn"

# Paths to the model and files
model_path = os.path.join(project_folder, 'results_interpro/model_epoch_8.pth')
file_go = os.path.join(project_folder, 'go_terms.list')
file_interpro = os.path.join(project_folder, 'interpro_terms.list')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"

# Load GO and InterPro terms
with open(file_go, 'r') as file:
    go_list = [line.strip() for line in file.readlines()]

with open(file_interpro, 'r') as file:
    interpro_list = [line.strip() for line in file.readlines()]

# Define the neural network model class
class SingleLayerNN(nn.Module):
    def __init__(self, input_size=15976, output_size=20181):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # Input to output

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # Apply sigmoid to output to ensure 0/1 probabilities
        return x

# Load the model
model = SingleLayerNN(input_size=len(interpro_list), output_size=len(go_list)).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

model.eval()  # Set model to evaluation mode

# Function to convert protein domains to input vector
def make_input(protein_domains):
    input_vector = [1 if term in protein_domains else 0 for term in interpro_list]
    X = np.asarray(input_vector)
    return X

# Function to print GO term information
def print_go(list_prediction):
    import requests

    def get_go_term_info(go_id):
        base_url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
        response = requests.get(base_url, headers={"Accept": "application/json"})

        if response.status_code == 200:
            data = response.json()
            term_info = data.get('results', [])[0]
            print(f"GO Term: {go_id}")
            print(f"Name: {term_info['name']}")
            print(f"Ontology: {term_info['aspect']}")
            print(f"Definition: {term_info['definition']['text']}\n")
        else:
            print(f"GO Term: {go_id} not found.\n")

    go_predicted = [term for term, pred in zip(go_list, list_prediction) if pred == 1]
    if len(go_predicted) == 0:
        print('No functions predicted')
    else:
        for go_id in go_predicted:
            get_go_term_info(go_id)

# Function to predict protein function based on the provided domains
def predict_protein_function(protein_domains_list):
    for idx, protein_domains in enumerate(protein_domains_list):
        # Create input tensor from the protein domains
        input_vector = make_input(protein_domains)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

        # Run the model to get predictions
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Convert predictions to binary outputs (0 or 1) using a threshold (e.g., 0.5)
        prediction_binary = (prediction > 0.5).int().cpu().numpy().flatten()
        
        print(f"=========================================")
        print(f"PREDICTIONs for Protein #{idx}\n")

        # Print GO predictions
        print_go(prediction_binary)
        
# Example usage:
# Replace this list with your actual protein domain data.
example_protein_domains_list = [
    ['IPR000031','IPR000039', 'IPR000046'],  # Example protein 1
    ['IPR000039', 'IPR000046'],  # Example protein 2
]

# Predict functions for the given protein domains
predict_protein_function(example_protein_domains_list)
