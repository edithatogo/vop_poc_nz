

file_path = "/Users/doughnut/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/Submitted/policy_societaldam_pharma/nzmj_feedback/methods_supplement_v7_20251128.md"

with open(file_path) as f:
    lines = f.readlines()

new_lines = []
in_table = False
header_processed = False

for line in lines:
    if line.startswith("| Parameter |"):
        in_table = True
        new_lines.append(line.strip() + " Reference |\n")
        continue

    if in_table and line.startswith("| :---"):
        new_lines.append(line.strip() + " :--- |\n")
        continue

    if in_table and line.startswith("|"):
        parts = line.split("|")
        # parts[0] is empty, parts[1] is Parameter, ...
        # Structure: | Parameter | Value | Description | Source | Perspective | intervention |
        # Indices: 0="", 1=Param, 2=Val, 3=Desc, 4=Source, 5=Persp, 6=Interv, 7=""

        if len(parts) >= 7:
            source = parts[4].strip()
            ref = "{See References}"

            if "Model specification" in source or "Model assumption" in source:
                ref = "{Author, 2025 @manuscript #1}"
            elif "Standard practice" in source:
                ref = "{PHARMAC, 2020 @pharmac2020 #13}"
            elif "Ministry of Health" in source:
                ref = "{Ministry of Health, 2023 @moh2023 #8}"
            elif "PHARMAC" in source:
                ref = "{PHARMAC, 2023 @pharmac2023 #31}"
            elif "Stats NZ" in source:
                ref = "{Stats NZ, 2023 @statsnz2023 #99}"
            elif "Literature" in source:
                ref = "{See References}"

            new_lines.append(line.strip() + f" {ref} |\n")
        else:
            new_lines.append(line)
    else:
        in_table = False
        new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
