import re
# Read the input QMDL log file
with open("/home/shreyasi_choudhary/Parser/data/data_july_6_30_75_120/data_with_qmdl_b881_v2_july_2.txt", "r") as infile:
    content = infile.read()

# Find all body entries using regex
bodies = re.findall(r"Body: b'(.*?)'", content)

# Write them to a new file in the desired format
with open("/home/shreyasi_choudhary/Parser/data/data_july_6_30_75_120/separated_body.txt", "w") as outfile:
    for body in bodies:
        outfile.write(f"Body: b'{body}'\n")

