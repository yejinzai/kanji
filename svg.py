import xml.etree.ElementTree as ET
import os
import cairosvg
import argparse


# 1. Parse the KanjiVG XML to extract paths for each kanji
def parse_kanjivg(kanjivg_file):
    tree = ET.parse(kanjivg_file)
    root = tree.getroot()

    # Dictionary to store kanji and their associated SVG paths
    kanji_paths = {}

    for kanji in root.findall('.//kanji'):
        kanji_id = kanji.attrib['id']  # Get kanji ID like "kvg:kanji_0611b"
        paths = []

        for path in kanji.findall('.//path'):
            path_data = path.attrib['d']  # Get SVG path 'd' attribute
            paths.append(path_data)

        kanji_paths[kanji_id] = paths

    return kanji_paths


# 2. Generate flat SVG and PNG files for specified or all kanji
def generate_flat_svg_and_png(kanji_paths, svg_output_dir, png_output_dir, specified_kanji_id=None):
    # Create output directories if they don't exist
    os.makedirs(svg_output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    for kanji_id, paths in kanji_paths.items():
        # If specified_kanji_id is provided, skip processing other kanji
        if specified_kanji_id and kanji_id != specified_kanji_id:
            continue

        # Create SVG content with default attributes for fill and stroke, and set viewBox to 128x128
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">\n'
        )
        for path in paths:
            svg_content += f'  <path d="{path}" fill="none" stroke="black" stroke-width="2"/>\n'
        svg_content += '</svg>'

        # Save the combined paths as an SVG file
        kanji_unicode = kanji_id.split('_')[-1]  # Extract Unicode part from "kvg:kanji_0611b"
        svg_file = os.path.join(svg_output_dir, f'{kanji_unicode}.svg')

        with open(svg_file, 'w') as f:
            f.write(svg_content)

        print(f"Generated SVG for kanji {kanji_unicode} at {svg_file}")

        # Convert SVG to PNG with 128x128 resolution
        png_file = os.path.join(png_output_dir, f'{kanji_unicode}.png')
        cairosvg.svg2png(url=svg_file, write_to=png_file, output_width=128, output_height=128)
        print(f"Converted {svg_file} to {png_file} with 128x128 resolution")


# Main function to orchestrate the SVG and PNG generation
def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description='Generate flat SVG and PNG files from KanjiVG XML data.')
    parser.add_argument('--kanji_id', type=str, default=None,
                        help='Optional: Kanji ID to process (e.g., "kvg:kanji_0611b").')

    # Parse the arguments
    args = parser.parse_args()

    kanjivg_file = "kanjivg-20220427.xml"
    svg_output_dir = "flat_kanji/svg"
    png_output_dir = "flat_kanji/png"

    # Step 1: Parse KanjiVG XML for kanji paths
    kanji_paths = parse_kanjivg(kanjivg_file)

    # Step 2: Generate flat SVGs and PNGs for specified or all kanji
    generate_flat_svg_and_png(kanji_paths, svg_output_dir, png_output_dir, args.kanji_id)


if __name__ == '__main__':
    main()
