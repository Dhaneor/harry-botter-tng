import subprocess
import re


def get_clean_requirements():
    # Get all installed packages
    result = subprocess.run(["pip", "freeze", "--all"], capture_output=True, text=True)
    packages = result.stdout.split('\n')

    clean_requirements = []
    for package in packages:
        # Skip empty lines
        if not package.strip():
            continue

        # Extract package name and version
        match = re.match(r'([^=\s]+)==([^=\s]+)', package)
        if match:
            package_name, version = match.groups()
            clean_requirements.append(f"{package_name}=={version}")
        else:
            # For packages without a version or with a different format,
            # try to get their version using pip show
            package_name = package.split('==')[0].split('@')[0].strip()
            pip_show = subprocess.run(
                ["pip", "show", package_name], capture_output=True, text=True
                )
            version_match = re.search(r'Version: (.+)', pip_show.stdout)
            if version_match:
                version = version_match.group(1)
                clean_requirements.append(f"{package_name}=={version}")

    return clean_requirements


# Generate and save clean requirements
clean_reqs = get_clean_requirements()
with open("requirements_mvp.txt", "w") as f:
    f.write("\n".join(clean_reqs))

print("Clean requirements.txt file has been generated.")
