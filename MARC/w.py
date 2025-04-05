import h5py
import numpy as np
import sys
import io

# Redirect stdout to capture potentially long output
old_stdout = sys.stdout
sys.stdout = captured_output = io.StringIO()

def inspect_h5_weights(filepath):
    """Prints layer names and weight shapes from an H5 weights file."""
    print(f"\n--- Inspecting Weights File: {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            # --- Try Keras H5 format first (has 'layer_names' attribute) ---
            if 'layer_names' in f.attrs:
                layer_names = [n.decode('utf-8') for n in f.attrs['layer_names']]
                print(f"Found {len(layer_names)} layers based on 'layer_names' attribute (Keras format):")
                for name in layer_names:
                    print(f"\nLayer: {name}")
                    try:
                        g = f[name]
                        weight_names = [n.decode('utf-8') for n in g.attrs['weight_names']]
                        print(f"  Weight Names: {weight_names}")
                        for weight_name in weight_names:
                            try:
                                # Keras saves weights typically as group/weight_name (e.g., dense_1/kernel:0)
                                # The actual dataset might be nested deeper or named slightly differently
                                # Let's try finding the dataset within the group 'g'
                                weight_data = None
                                if weight_name in g: # Direct match in group
                                    weight_data = g[weight_name]
                                else:
                                    # Check for common patterns like 'kernel:0' -> 'kernel'
                                    simple_name_key = weight_name.split('/')[-1].split(':')[0] # e.g., 'kernel'
                                    if simple_name_key in g:
                                         weight_data = g[simple_name_key]


                                if weight_data is not None and hasattr(weight_data, 'shape'):
                                    print(f"  - {weight_name}: Shape = {weight_data.shape}, Dtype = {weight_data.dtype}")
                                else:
                                     print(f"  - {weight_name}: Weight data not found directly within group '{name}'. Items: {list(g.keys())}")

                            except Exception as e_w:
                                print(f"    Error accessing weight {weight_name}: {e_w}")
                                print(f"    Items in group '{name}': {list(g.keys())}")
                    except Exception as e:
                        print(f"  Error accessing details for layer {name}: {e}")


            # --- If not Keras format, try TensorFlow format (top-level groups are layers) ---
            else:
                print("No 'layer_names' attribute found. Assuming TensorFlow H5 format (top-level groups are layers).")
                top_level_groups = list(f.keys())
                print(f"Found {len(top_level_groups)} top-level groups: {top_level_groups}")
                layer_count_with_weights = 0
                for group_name in top_level_groups:
                    print(f"\nGroup/Layer: {group_name}")
                    try:
                        g = f[group_name]
                        # In TF format, weights are often under a sub-group named after the layer again, or 'vars'
                        weights_found = False
                        possible_weight_locations = [g]
                        if group_name in g and isinstance(g[group_name], h5py.Group):
                            possible_weight_locations.append(g[group_name])
                        if 'vars' in g and isinstance(g['vars'], h5py.Group):
                            possible_weight_locations.append(g['vars'])


                        datasets_in_group = {}
                        def find_datasets(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                datasets_in_group[name] = obj

                        for loc in possible_weight_locations:
                            loc.visititems(find_datasets)


                        if datasets_in_group:
                            weights_found = True
                            print(f"  Weights found:")
                            for name, dset in datasets_in_group.items():
                                 print(f"  - {name}: Shape = {dset.shape}, Dtype = {dset.dtype}")
                        else:
                            print(f"  No weights (datasets) found directly under group '{group_name}' or common sub-groups.")

                        if weights_found:
                            layer_count_with_weights += 1

                    except Exception as e:
                        print(f"  Error accessing details for group {group_name}: {e}")
                print(f"\nTotal groups/layers containing weights found: {layer_count_with_weights}")

    except Exception as e:
        print(f"FATAL ERROR: Could not open or read H5 file {filepath}. Error: {e}")

    print("--- Inspection Complete ---")

# --- Usage ---
weights_path = './MARC/MARC_Weight.h5'
inspect_h5_weights(weights_path)

# Restore stdout and print captured output
sys.stdout = old_stdout
output_string = captured_output.getvalue()
print(output_string)

# Optionally write to a file if output is very long
# with open("h5_inspection_output.txt", "w") as f:
#     f.write(output_string)
# print("Inspection output also saved to h5_inspection_output.txt")