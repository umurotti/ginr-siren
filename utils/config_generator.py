import itertools
import yaml
import sys
from collections import OrderedDict
import os

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def save_config(config, filename):
    with open(filename, 'w') as file:
        ordered_dump(config, file, Dumper=yaml.SafeDumper)

def input_value(prompt, value_type):
    while True:
        user_input = input(prompt)
        if not user_input:
            return None

        if value_type == 'string':
            return user_input
        elif value_type == 'integer':
            try:
                return int(user_input)
            except ValueError:
                print("Invalid input, please enter an integer.")
        elif value_type == 'list':
            return [item.strip() for item in user_input.split(',')]
        else:
            return user_input
        
def generate_combinations(experiment_data):
    keys = experiment_data.keys()
    # Wrap non-iterable values in a list
    values_product = itertools.product(
        *([v] if not isinstance(v, list) else v for v in experiment_data.values())
    )
    return keys, values_product

def convert_to_values(d):
    for key, value in list(d.items()):
        if isinstance(value, dict) and 'value' in value:
            d[key] = value['value']
        elif isinstance(value, dict):
            convert_to_values(value)
            
def update_config(config, key, value):
    key_parts = key.split('.')
    sub_config = config
    for part in key_parts[:-1]:
        if part not in sub_config:
            sub_config[part] = {}
        sub_config = sub_config[part]
    if isinstance(sub_config.get(key_parts[-1], None), dict):
        sub_config[key_parts[-1]]['value'] = value
    else:
        sub_config[key_parts[-1]] = {'value': value, 'type': 'string'}  # default type can be adjusted
        
def format_key_value_for_filename(key, value):
    # Formats a key-value pair into a string suitable for filenames
    key_formatted = key.replace('.', '_')

    # If the value is a file path, extract just the file name
    if isinstance(value, str) and os.path.isfile(value):
        value = os.path.basename(value)

    # Replace any other disallowed characters in file names
    value = str(value).replace('/', '_').replace('\\', '_')

    if isinstance(value, list):
        value_formatted = '-'.join(map(str, value))
    else:
        value_formatted = value

    return f"{key_formatted}={value_formatted}"


# Function to identify keys with multiple values
def keys_with_multiple_values(experiment_data):
    return {key for key, value in experiment_data.items() if isinstance(value, list) and len(value) > 1}

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <template_file_path> <output_file_path>")
        #sys.exit(1)

    default_template_file = './config/template_config_relu.yaml'
    default_output_file = './config/res_config.yaml'
    default_use_experiment_file = True

    template_file = sys.argv[1] if len(sys.argv) > 1 else default_template_file
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output_file
    use_experiment_file = sys.argv[3] if len(sys.argv) > 3 else default_use_experiment_file
    
    if not use_experiment_file:
        config = ordered_load(open(template_file, 'r'))

        keys_to_edit = input('Enter comma-separated keys to edit (e.g., dataset.supervision, arch.rank): ')
        keys_to_edit = [key.strip() for key in keys_to_edit.split(',')]

        def prompt_for_values(d, parent_key=''):
            for key, value_info in d.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value_info, dict) and 'value' in value_info and 'type' in value_info:
                    if full_key in keys_to_edit:
                        prompt = f'Enter value for {full_key} (current: {value_info["value"]}): '
                        d[key]['value'] = input_value(prompt, value_info['type'])
                elif isinstance(value_info, dict):
                    prompt_for_values(value_info, full_key)

        prompt_for_values(config)

        convert_to_values(config)

        save_config(config, output_file)
        print(f"Configuration file saved as {output_file}")
    
    else:
        #experiment_file = input("Enter the path to the experiment file: ")
        experiment_file = "config/final/onceki_ckpt/val_config_generator.yaml"
        experiments = ordered_load(open(experiment_file, 'r'))
        
        # Path for the parent "experiments" directory
        experiments_dir = os.path.join(os.getcwd(), 'experiments')
        os.makedirs(experiments_dir, exist_ok=True)
        
        for experiment in experiments:
            experiment_name = experiment['experiment_name']
            experiment_data = {k: v for k, v in experiment.items() if k != 'experiment_name'}
            
            # Create a directory for the experiment
            experiment_dir = os.path.join(experiments_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Identify which keys have multiple values
            varying_keys = keys_with_multiple_values(experiment_data)
            
            keys, combinations = generate_combinations(experiment_data)
            combo_index = 1
            
            for combo in combinations:
                config = ordered_load(open(template_file, 'r'))
                
                # Update config with the combination values and prepare filename parts
                filename_parts = []
                for key, value in zip(keys, combo):
                    update_config(config, key, value)
                    if key in varying_keys:
                        filename_part = format_key_value_for_filename(key, value)
                        filename_parts.append(filename_part)
                
                # Convert to final values
                convert_to_values(config)
                
                # Join the filename parts and limit the length if necessary
                filename_str = '_'.join(filename_parts)
                # Limit the filename length to a reasonable number if needed
                max_filename_length = 255  # Adjust as needed
                if len(filename_str) > max_filename_length:
                    filename_str = filename_str[:max_filename_length]
                
                # Generate a filename that reflects the combination
                output_filename = f"{experiment_name}_config_{filename_str}.yaml"
                output_path = os.path.join(experiment_dir, output_filename)
                save_config(config, output_path)
                print(f"Configuration file saved as {output_path}")
                combo_index += 1
                        


if __name__ == "__main__":
    main()
