def add_autosummary_to_modules(rst_file_path):
    with open(rst_file_path, "r") as file:
        content = file.readlines()

    # Prepare the autosummary content
    autosummary_content = []
    collecting = False  # Flag to start collecting module names from toctree

    for line in content:
        if ".. toctree::" in line:
            collecting = True  # Start collecting after finding toctree
        elif collecting:
            if line.strip() and not line.strip().startswith(":"):
                module_name = line.strip()
                autosummary_content.append(module_name)
            # else:
                collecting = False  # Stop collecting if line is empty or toctree options are done
    
    autosummary_block = ".. autosummary::\n\t:toctree: _autosummary\n\t:nosignatures:\n\n"
    # Generate the autosummary block if there are modules to summarize
    if autosummary_content:
        
        for module in autosummary_content:
            autosummary_block += f"\t{module}\n"

    # Writing the content back to the file with the autosummary inserted
    with open(rst_file_path, "w") as file:
        # header_written = False
        for line in content:
            if "=" not in line:
                file.write(line)
            elif line.strip().endswith('='):
                file.write(line)
                # Insert autosummary block right after the section header
                file.write(autosummary_block + "\n")


# Example usage
rst_file_path = './source/apis/components/components.agent.rst'
add_autosummary_to_modules(rst_file_path)
