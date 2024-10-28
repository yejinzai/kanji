import xml.etree.ElementTree as ET
import os
import pandas as pd

# 1. Parse KANJIDIC2 XML to extract kanji and consolidate their English meanings
def parse_kanjidic2(file):
    tree = ET.parse(file)
    root = tree.getroot()
    kanji_meanings = {}

    for character in root.findall('character'):
        kanji = character.find('literal').text
        meanings = [m.text for m in character.findall('.//meaning') if 'm_lang' not in m.attrib]

        if kanji in kanji_meanings:
            # If kanji is already in the dictionary, extend the meanings list
            kanji_meanings[kanji].extend(meanings)
        else:
            # Otherwise, create a new entry for the kanji
            kanji_meanings[kanji] = meanings

    # Consolidate meanings into a single, comma-separated string for each kanji
    consolidated_kanji_meanings = [(k, ";".join(set(v))) for k, v in kanji_meanings.items()]

    return consolidated_kanji_meanings

# 2. Match kanji to their corresponding SVG and PNG file paths
def map_kanji_to_files(kanji_meanings, svg_dir, png_dir):
    svg_paths = []
    png_paths = []
    meanings_list = []
    kanji_list = []

    for kanji, meanings in kanji_meanings:
        # Convert kanji to its Unicode code point in lowercase hexadecimal
        unicode_code_point = f'{ord(kanji):05x}'

        # Construct expected SVG and PNG file paths based on the naming convention
        svg_file = os.path.join(svg_dir, f'{unicode_code_point}.svg')
        png_file = os.path.join(png_dir, f'{unicode_code_point}.png')

        if os.path.exists(svg_file) and os.path.exists(png_file):
            # Append to lists for metadata
            svg_paths.append(svg_file)
            png_paths.append(png_file)
            meanings_list.append(meanings)
            kanji_list.append(kanji)
        else:
            print(f"Files not found for kanji: {kanji} (Unicode: {unicode_code_point})")

    return kanji_list, svg_paths, png_paths, meanings_list

# 3. Create a CSV metadata file
def create_metadata(kanji, svg_paths, png_paths, meanings, output_csv):
    data = {
        'kanji': kanji,
        'meanings': meanings,
        'svg_path': svg_paths,
        'png_path': png_paths
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

# Main function to orchestrate the dataset preparation
def main():
    # Paths to KANJIDIC2 XML, SVG, and PNG directories
    kanjidic2_file = 'kanjidic2.xml'
    svg_dir = 'flat_kanji/svg'
    png_dir = 'flat_kanji/png'
    output_csv = 'flat_kanji/metadata.csv'

    # Step 1: Parse KANJIDIC2 for kanji and meanings
    kanji_meanings = parse_kanjidic2(kanjidic2_file)

    # Step 2: Map kanji to existing SVG and PNG file paths
    kanji_list, svg_paths, png_paths, meanings_list = map_kanji_to_files(
        kanji_meanings, svg_dir, png_dir
    )

    # Step 3: Create CSV metadata file
    create_metadata(kanji_list, svg_paths, png_paths, meanings_list, output_csv)
    print(f"Metadata CSV created at '{output_csv}'.")

if __name__ == '__main__':
    main()