import os
import json
import argparse
import torch
from transformers import AutoTokenizer, pipeline
from utils import get_device_config, extract_code, format_code

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate demo outputs for SpecialtyTuner")
    parser.add_argument("--output_file", default="demo_outputs.md", help="Path to save demo outputs")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()
    
    # Check if model exists
    checkpoint_path = "checkpoints/best"
    if not os.path.exists(checkpoint_path):
        print("Model checkpoint not found. Running demo in simulation mode.")
        # Generate fake outputs for demonstration purposes
        simulate_outputs(args.output_file)
        return
    
    # Load test cases - use a set of high-quality examples for demo
    demo_examples = [
        "Load a CSV file with pandas and handle missing values",
        "Train a random forest classifier with scikit-learn and evaluate with cross-validation",
        "Perform k-means clustering and visualize the results with 3D plot",
        "Create a convolutional neural network with PyTorch for image classification",
        "Implement a sentiment analysis model using TF-IDF and logistic regression",
        "Generate synthetic data for regression using scikit-learn and visualize with seaborn",
        "Implement a simple LSTM network for sequence prediction"
    ]
    
    if os.path.exists("data/test_cases.json"):
        with open("data/test_cases.json", "r") as f:
            custom_test_cases = json.load(f)
            # Add any custom examples but keep the total reasonable
            all_cases = demo_examples + custom_test_cases
            test_cases = all_cases[:7]
    else:
        test_cases = demo_examples
    
    # Determine device
    device, torch_dtype, _, device_name = get_device_config()
    print(f"Using {device_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Create text generation pipeline
    model = pipeline(
        "text-generation",
        model=checkpoint_path,
        tokenizer=tokenizer,
        device=device,
        max_length=1024  # Increased max length for more complete examples
    )
    
    # Generate outputs
    generate_outputs(model, test_cases, args.output_file, temperature=args.temperature)

def generate_outputs(model, test_cases, output_file, temperature=0.2):
    """Generate code for each test case and save to file"""
    outputs = []
    
    for prompt in test_cases:
        print(f"Generating code for: {prompt}")
        full_prompt = f"Write Python ML code for: {prompt}\nMake sure the code is complete, well-documented, and follows best practices."
        
        # Generate code with improved parameters
        outputs_list = model(
            full_prompt, 
            temperature=temperature,
            max_length=1024,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        output_text = outputs_list[0]['generated_text']
        
        # Extract and format code
        code = extract_code(output_text, full_prompt)
        formatted_code = format_code(code)
        
        outputs.append({
            "prompt": prompt,
            "code": formatted_code
        })
    
    # Save to markdown file
    with open(output_file, "w") as f:
        f.write("# SpecialtyTuner Demo Outputs\n\n")
        for item in outputs:
            f.write(f"## {item['prompt']}\n\n")
            f.write("```python\n")
            f.write(item['code'])
            f.write("\n```\n\n")
    
    print(f"Demo outputs saved to {output_file}")

def simulate_outputs(output_file):
    """Generate simulated outputs for demo purposes"""
    simulated_outputs = [
        {
            "prompt": "Load a CSV file with pandas",
            "code": "import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('data.csv')\n\n# Display the first few rows\nprint(df.head())"
        },
        {
            "prompt": "Train a logistic regression model",
            "code": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# Prepare data\nX = df.drop('target', axis=1)\ny = df['target']\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Train model\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\n\n# Evaluate\npredictions = model.predict(X_test)\naccuracy = accuracy_score(y_test, predictions)\nprint(f'Accuracy: {accuracy:.2f}')"
        },
        {
            "prompt": "Perform k-means clustering",
            "code": "from sklearn.cluster import KMeans\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Assuming X contains your features\n\n# Determine optimal number of clusters using elbow method\ndistortions = []\nK_range = range(1, 10)\nfor k in K_range:\n    kmeans = KMeans(n_clusters=k, random_state=42)\n    kmeans.fit(X)\n    distortions.append(kmeans.inertia_)\n\n# Plot elbow curve\nplt.figure(figsize=(10, 6))\nplt.plot(K_range, distortions, 'bx-')\nplt.xlabel('Number of clusters')\nplt.ylabel('Distortion')\nplt.title('Elbow Method For Optimal k')\nplt.show()\n\n# Apply K-means with optimal k\noptimal_k = 4  # Choose based on elbow curve\nkmeans = KMeans(n_clusters=optimal_k, random_state=42)\ncluster_labels = kmeans.fit_predict(X)\n\n# Get cluster centers\ncenters = kmeans.cluster_centers_"
        },
        {
            "prompt": "Create a simple neural network with PyTorch",
            "code": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Define a simple neural network\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNN, self).__init__()\n        self.layer1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.layer2 = nn.Linear(hidden_size, output_size)\n    \n    def forward(self, x):\n        x = self.layer1(x)\n        x = self.relu(x)\n        x = self.layer2(x)\n        return x\n\n# Initialize the model\ninput_size = 10\nhidden_size = 20\noutput_size = 2\nmodel = SimpleNN(input_size, hidden_size, output_size)\n\n# Define loss function and optimizer\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\n\n# Example training loop\nnum_epochs = 10\nfor epoch in range(num_epochs):\n    # Forward pass\n    outputs = model(inputs)\n    loss = criterion(outputs, targets)\n    \n    # Backward and optimize\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()\n    \n    if (epoch+1) % 2 == 0:\n        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')"
        },
        {
            "prompt": "Handle missing data in a DataFrame",
            "code": "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Assuming df is your DataFrame with missing values\n\n# Check for missing values\nprint('Missing values per column:')\nprint(df.isnull().sum())\n\n# Visualize missing values\nplt.figure(figsize=(10, 6))\nsns.heatmap(df.isnull(), cbar=False, cmap='viridis')\nplt.title('Missing Value Heatmap')\nplt.show()\n\n# Handle missing values\n# 1. Fill numerical columns with mean\nnum_cols = df.select_dtypes(include=[np.number]).columns\nfor col in num_cols:\n    df[col].fillna(df[col].mean(), inplace=True)\n\n# 2. Fill categorical columns with mode\ncat_cols = df.select_dtypes(include=['object']).columns\nfor col in cat_cols:\n    df[col].fillna(df[col].mode()[0], inplace=True)\n\n# Verify missing values are handled\nprint('\\nRemaining missing values:')\nprint(df.isnull().sum())"
        }
    ]
    
    # Save to markdown file
    with open(output_file, "w") as f:
        f.write("# SpecialtyTuner Demo Outputs (Simulated)\n\n")
        f.write("> Note: These are simulated outputs since the model hasn't been trained yet.\n\n")
        for item in simulated_outputs:
            f.write(f"## {item['prompt']}\n\n")
            f.write("```python\n")
            f.write(item['code'])
            f.write("\n```\n\n")
    
    print(f"Simulated demo outputs saved to {output_file}")

if __name__ == "__main__":
    main() 