from PIL import Image
import os

def ascii_to_image(ascii_file_path: str, output_color: str = 'L'):
    """
    Converts an ASCII art text file back into a grayscale image.

    :param ascii_file_path: The path to the input ASCII text file (e.g., 'beautiful_woman_ascii_art.txt').
    :param output_color: The color mode for the output image. 'L' for grayscale (default).
    """
    
    # --- 1. Define the Reverse Mapping ---
    # Must use the same character set as the conversion process!
    ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'] 
    
    # Calculate the step size (how much brightness each character represents)
    step_size = 256 / len(ascii_chars)
    
    # Create the reverse map: character -> pixel value (0-255)
    reverse_map = {}
    for i, char in enumerate(ascii_chars):
        # We map the index back to a brightness value. We use the midpoint of the step for the value.
        pixel_value = int(i * step_size + step_size / 2)
        reverse_map[char] = pixel_value
        
    # --- 2. Read the ASCII Art and Determine Dimensions ---
    try:
        with open(ascii_file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()] # Read lines, strip newlines, ignore empty lines
    except FileNotFoundError:
        print(f"Error: ASCII text file not found at '{ascii_file_path}'")
        return
    
    if not lines:
        print("Error: The text file is empty or contains no readable ASCII art.")
        return

    # Dimensions
    image_height = len(lines)
    image_width = len(lines[0])
    
    print(f"Reading file. Detected image dimensions: {image_width} wide x {image_height} high.")
    
    # --- 3. Create a New Image and Pixel Map ---
    # Create a new image in 'L' (8-bit grayscale) mode
    new_image = Image.new(output_color, (image_width, image_height))
    pixels = new_image.load()

    # Iterate through the lines and characters to set pixel values
    for y, line in enumerate(lines):
        # We iterate up to the determined width to handle any trailing spaces or uneven lines
        for x in range(image_width):
            # Ensure we don't go out of bounds if a line is shorter than the first one
            if x < len(line):
                char = line[x]
                # Look up the corresponding pixel value
                pixel_value = reverse_map.get(char, 0) # Default to 0 (black) if an unknown character is found
            else:
                # If the line is shorter, treat the remaining part as ' ' (space/white)
                pixel_value = reverse_map.get(' ', 255)
            
            # Set the pixel at (x, y) to the brightness value
            pixels[x, y] = pixel_value

    # --- 4. Save the Image ---
    # Create the output filename based on the input filename
    base_name = os.path.splitext(os.path.basename(ascii_file_path))[0]
    output_filename = f"{base_name}_restored.png"

    # Save as PNG, which is good for lossless grayscale images
    new_image.save(output_filename)

    print(f"✅ Conversion complete! Image restored and saved to '{output_filename}'")


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Adjust this variable to your ASCII file!
    input_ascii_file = "beautiful_woman_ascii_art.txt" 
    
    # NOTE: You would typically run the 'image_to_ascii' from the previous turn 
    # to generate this file first!
    
    # If the file exists from the previous run, uncomment and execute the line below:
    # ascii_to_image(input_ascii_file)
    
    print("\nTo run the restoration, make sure you have an ASCII file ready and uncomment the last line in the 'if __name__ == '__main__': block.")
