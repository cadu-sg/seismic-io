from os import path, getcwd

version_file_path = path.join(getcwd(), "seismicio/constants/__version__.py")

new_release_version = ""

with open(version_file_path, 'r') as file:
  version_numerators = []
  last_release_version = file.read()
  version_constant_declaration = last_release_version.split("'")[0]
  version_numerators = last_release_version.replace("'", "").split(".")

  version_numerators[0] = version_numerators[0][len(version_numerators[0]) - 1]
  version_numerators[1] = int(version_numerators[1])
  version_numerators[2] = int(version_numerators[2]) + 1

  # concat numerators 
  new_release_version = f"{version_numerators[0]}.{version_numerators[1]}.{version_numerators[2]}"
  # contat to full __version__ declaration
  new_release_version = f"{version_constant_declaration}'{new_release_version}'"

with open(version_file_path, 'w') as file:
  file.write(new_release_version)
  print(new_release_version)
