import csv
import matplotlib.pyplot as plt

def parse_network_config(line):
    """Parse the network configuration line"""
    # Remove '# configuration: ' prefix and split into parts
    config = line.replace('# configuration:', '').strip()
    # Split into topology and ensemble info
    topology_str, ensemble_info = config.split('n=')
    
    # Parse topology (convert string tuple to list of integers)
    topology = eval(topology_str.strip())
    topology = list(topology)
    
    # Parse number of models in ensemble
    n_models = int(ensemble_info.replace('ensemble', '').strip())
    
    return topology, n_models

def parse_epoch_data(line):
    """Parse a line containing epoch and ratio data"""
    # Remove 'n_epochs = ' and split into epoch and ratio
    parts = line.replace('n_epochs =', '').split(',')
    epoch = int(parts[0].strip())
    ratio = float(parts[1].split('=')[1].strip())
    return epoch, ratio

def read_config_file(filename):
    """Read and parse the configuration file"""
    config_data = {
        'topology': None,
        'n_models': None,
        'epochs_data': []
    }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Parse first line for network configuration
        if lines and lines[0].startswith('# configuration:'):
            config_data['topology'], config_data['n_models'] = parse_network_config(lines[0])
        
        # Parse remaining lines for epoch data
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                epoch, ratio = parse_epoch_data(line)
                config_data['epochs_data'].append((epoch, ratio))
    
    return config_data

def write_to_csv(config_data, output_filename):
    """Write the configuration data to a CSV file"""
    # Convert topology list to string representation
    topology_str = '_'.join(map(str, config_data['topology']))
    
    # Prepare the CSV data
    headers = ['Topology', 'n_models_ensemble', 'n_epoch', 'ratio_mse']
    rows = []
    
    # Create a row for each epoch data point
    for epoch, ratio in config_data['epochs_data']:
        row = [
            topology_str,
            config_data['n_models'],
            epoch,
            ratio
        ]
        rows.append(row)
    
    # Write to CSV file
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def plot_results(config_data):
    """Create a plot of epochs vs ratio with a horizontal line at y=1"""
    epochs = [epoch for epoch, _ in config_data['epochs_data']]
    ratios = [ratio for _, ratio in config_data['epochs_data']]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.plot(epochs, ratios, 'bo-', label='MSE Ratio')
    
    # Add horizontal dashed line at y=1
    plt.axhline(y=1, color='r', linestyle='--', label='Baseline (y=1)')
    
    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('MSE Ratio')
    topology_str = '_'.join(map(str, config_data['topology']))
    plt.title(f'MSE Ratio vs Epochs\nTopology: {topology_str}, Ensemble Size: {config_data["n_models"]}')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('ratio_plot.png')
    plt.close()

if __name__ == "__main__":
    input_filename = "results.txt"
    output_filename = "results.csv"
    
    try:
        # Read the configuration file
        config = read_config_file(input_filename)
        
        # Write to CSV
        write_to_csv(config, output_filename)
        print(f"Successfully wrote results to {output_filename}")
        
        # Create and save the plot
        plot_results(config)
        print("Successfully created plot: ratio_plot.png")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_filename}'")
    except Exception as e:
        print(f"Error: {str(e)}") 