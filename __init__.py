from nodes import interrupt_processing
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import random

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")
loop_interation = 0
reset_global = False
class BookToolsPromptSelector:
    """
    Selects and concatenates values from a dictionary based on input indices and ranges.
    The input string contains 1-based indices which are converted to dictionary keys.
    Assumes the dictionary values are strings which will be concatenated and returned as a single string.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Describes expected input types:
        - 'dictionary': a dictionary where keys will be selected based on 'selected_indexes'.
        - 'selected_indexes': a int listing keys by index, separated by commas, with optional ranges.
        """
        return {
            "required": {
                "dictionary": ("DICTIONARY", ),
                "selected_indexes": ("INT", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "main"
    CATEGORY = "selector"
    OUTPUT_NODE = False

    def main(self, dictionary: dict, selected_indexes: int):
        if not isinstance(dictionary, dict):
            raise ValueError("Conditioning must be a dictionary.")

        if not isinstance(selected_indexes, int):
            raise ValueError("selected_indexes must be an integer.")
        
        result_string = str(dictionary.get(str(selected_indexes), ''))
        
        return (result_string,)

class BookToolsPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":"\"first page description\",\n\"second page description\""}),
            "before_text": ("STRING", {"multiline": True,}),
            "after_text": ("STRING", {"multiline": True,}),
            }}

    RETURN_TYPES = ("DICTIONARY",)
    FUNCTION = "main"
    CATEGORY = "prompt"
    OUTPUT_NODE = False

    def main(self, text, before_text, after_text):
        """
        Splits the text on commas, applies before and after text to each segment,
        and returns a dictionary with each formatted text indexed.

        Parameters:
        - text (str): Comma-separated text entries to be formatted.
        - before_text (str): Text to prepend to each entry.
        - after_text (str): Text to append to each entry.

        Returns:
        - dict: Dictionary with indexed formatted texts.
        """
        segments = text.split('",')
        formatted_texts = {}

        for index, segment in enumerate(segments, start=1):
            cleaned_segment = segment.replace('"', '').strip()
            parts = [part for part in [before_text.strip(), cleaned_segment, after_text.strip()] if part]
            formatted_text = ', '.join(parts)
            formatted_texts[str(index)] = formatted_text
        return (formatted_texts,)

# based on https://civitai.com/models/26836/comfyui-loopback-nodes but take any parametrs and have reset option
class BookToolsLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"reset": ("BOOLEAN", {"default": False}),}}

    RETURN_TYPES = ("LOOP", "INT",)
    FUNCTION = "run"
    CATEGORY = "loopback"
    RETURN_NAMES = ("LOOP", "Iteration",)

    def run(self, reset):
        global loop_interation, reset_global
        if (reset == True):
            loop_interation = 1
            reset_global = True
        else:
            reset_global = False
        return (self, loop_interation)

    @classmethod
    def IS_CHANGED(self, reset):
        global loop_interation, reset_global
        loop_interation +=1
        if (reset == True):
            loop_interation = 1
            reset_global = True
        else:
            reset_global = False
        return loop_interation
                   
class BookToolsLoopStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "first_loop": (any,),
            "loop": ("LOOP",)
            }
        }

    FUNCTION = "run"
    CATEGORY = "loopback"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("*",)

    def run(self, first_loop, loop):
        global reset_global
        if reset_global == True:
            return (first_loop,)
        if hasattr(loop, 'next'):
            return (loop.next,)
        return (first_loop,)

    @classmethod
    def IS_CHANGED(self, first_loop, loop):
        global reset_global
        if reset_global == True:
            return (first_loop,)
        if hasattr(loop, 'next'):
            return id(loop.next)
        return float("NaN")

class BookToolsLoopEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "send_to_next_loop": (any, ), "loop": ("LOOP",) }}
    FUNCTION = "run"
    CATEGORY = "loopback"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def run(self, send_to_next_loop, loop):
        loop.next = send_to_next_loop
        return ()

class BookToolsEndQueue:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "boolean": ("BOOLEAN", {"default": False}), }}
    FUNCTION = "main"
    CATEGORY = "logic"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def main(self, boolean):
        if (boolean==True):
            interrupt_processing()
        return ()

class BookToolsDownloadFont:
    # Known working fonts from Google Fonts
    SUPPORTED_FONTS = {
        "Delius": "Delius:wght@400",
        "DejaVuSans-Bold": "DejaVuSans-Bold",
        "FreeMono": "FreeMono:wght@400",
        "Roboto": "RobotoMono-VariableFont_wght",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "font_name": (list(cls.SUPPORTED_FONTS.keys()), {"default": "Open Sans"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_font"
    CATEGORY = "image/text"

    def download_font(self, font_name):
        import requests
        import os
        from urllib.parse import urlparse
        
        # Default fallback font
        fallback_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        # Create fonts directory if it doesn't exist
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        os.makedirs(fonts_dir, exist_ok=True)
        
        # Use actual font filename instead of key
        font_filename = self.SUPPORTED_FONTS[font_name] + ".ttf"
        font_path = os.path.join(fonts_dir, font_filename)
        
        # If font already exists, verify it can be loaded
        if os.path.exists(font_path):
            try:
                ImageFont.truetype(font_path, 12)  # Test load with small size
                return (font_path,)
            except Exception:
                print(f"Cached font {font_name} is corrupt, attempting redownload...")
                os.remove(font_path)
        
        # Download font from Google Fonts
        api_url = f"https://fonts.googleapis.com/css2?family={self.SUPPORTED_FONTS[font_name]}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Get the CSS
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            css = response.text
            
            # Extract the font URL using more robust parsing
            for line in css.split('\n'):
                if 'src:' in line and 'url(' in line:
                    start = line.find('url(') + 4
                    end = line.find(')', start)
                    font_url = line[start:end].strip()
                    break
            else:
                raise ValueError("Could not find font URL in CSS")
            
            # Validate URL
            parsed_url = urlparse(font_url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Invalid font URL")
            
            # Download the font file
            font_response = requests.get(font_url, headers=headers)
            font_response.raise_for_status()
            
            # Save the font file
            with open(font_path, 'wb') as f:
                f.write(font_response.content)
            
            # Verify the downloaded font
            try:
                ImageFont.truetype(font_path, 12)
                return (font_path,)
            except Exception as e:
                print(f"Downloaded font {font_name} is invalid: {str(e)}")
                if os.path.exists(font_path):
                    os.remove(font_path)
                raise
                
        except Exception as e:
            print(f"Error downloading font {font_name}: {str(e)}")
            print(f"Falling back to default font: {fallback_font}")
            return (fallback_font,)

class BookToolsImageTextOverlay:
    def __init__(self, device="cpu"):
        self.device = device
    _alignments = ["left", "right", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "textbox_width": ("INT", {"default": 200, "min": 1}),  
                "textbox_height": ("INT", {"default": 200, "min": 1}),  
                "max_font_size": ("INT", {"default": 80, "min": 1, "max": 256, "step": 1}),
                "min_font_size": ("INT", {"default": 12, "min": 1, "max": 256, "step": 1}),
                "font": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}), 
                "alignment": (cls._alignments, {"default": "center"}),  
                "color": ("STRING", {"default": "#000000"}),  
                "start_x": ("INT", {"default": 0}),  
                "start_y": ("INT", {"default": 0}),
                "padding": ("INT", {"default": 50}),
                "line_height_factor": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_text_overlay"
    CATEGORY = "image/text"

    def calculate_text_size(self, text, font_size, font_path, max_width, max_height):
        font = ImageFont.truetype(font_path, font_size)
        paragraphs = text.split('\n')
        lines = []
        
        for paragraph in paragraphs:
            words = paragraph.split()
            current_line = []
            current_width = 0

            for word in words:
                word_bbox = font.getbbox(word)
                word_width = word_bbox[2] - word_bbox[0]
                space_width = font.getbbox(' ')[2] if current_line else 0
                
                if current_width + word_width + space_width <= max_width:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width

            if current_line:
                lines.append(' '.join(current_line))

        total_height = len(lines) * (font_size * 1.2)
        return lines, total_height <= max_height

    def add_text_overlay(self, image, text, textbox_width, textbox_height, max_font_size, min_font_size, 
                        font, alignment, color, start_x, start_y, padding, line_height_factor):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image_pil = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        effective_width = textbox_width - 2 * padding
        effective_height = textbox_height - 2 * padding

        # Binary search for the optimal font size
        low, high = min_font_size, max_font_size
        optimal_font_size = min_font_size
        optimal_lines = []

        while low <= high:
            mid = (low + high) // 2
            lines, fits = self.calculate_text_size(text, mid, font, effective_width, effective_height)
            
            if fits:
                optimal_font_size = mid
                optimal_lines = lines
                low = mid + 1
            else:
                high = mid - 1

        # If no fitting size was found, use min_font_size
        if not optimal_lines:
            optimal_font_size = min_font_size
            optimal_lines, _ = self.calculate_text_size(text, min_font_size, font, effective_width, effective_height)

        # Render the text with the optimal font size
        loaded_font = ImageFont.truetype(font, optimal_font_size)
        draw = ImageDraw.Draw(image_pil)
        
        line_height = int(optimal_font_size * line_height_factor)
        total_text_height = len(optimal_lines) * line_height
        y = start_y + padding

        # If text fits in height, center it vertically
        if total_text_height <= effective_height:
            y += (effective_height - total_text_height) // 2

        # Draw all lines, even if they overflow
        for line in optimal_lines:
            if y + line_height > start_y + textbox_height:  # Skip lines that would be completely outside the box
                break

            line_bbox = loaded_font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]

            if alignment == "left":
                x = start_x + padding
            elif alignment == "right":
                x = start_x + effective_width - line_width + padding
            else:  # center
                x = start_x + padding + (effective_width - line_width) // 2

            draw.text((x, y), line, fill=color_rgb, font=loaded_font)
            y += line_height

        image_tensor_out = torch.tensor(np.array(image_pil).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        return (image_tensor_out,)

class BookToolsCalculateTextGrowth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "mask_width": ("INT", {"default": 200, "min": 1}),
                "mask_height": ("INT", {"default": 200, "min": 1}),
                "min_font_size": ("INT", {"default": 12, "min": 1, "max": 256, "step": 1}),
                "font": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}),
                "padding": ("INT", {"default": 50}),
                "line_height_factor": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("growth_percentage",)
    FUNCTION = "calculate"
    CATEGORY = "image/text"

    def calculate_text_bounds(self, text, font_size, font_path, max_width, line_height_factor):
        font = ImageFont.truetype(font_path, font_size)
        paragraphs = text.split('\n')
        max_line_width = 0
        total_lines = 0
        
        for paragraph in paragraphs:
            words = paragraph.split()
            current_line = []
            current_width = 0

            for word in words:
                word_bbox = font.getbbox(word)
                word_width = word_bbox[2] - word_bbox[0]
                space_width = font.getbbox(' ')[2] if current_line else 0
                
                if current_width + word_width + space_width <= max_width:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    max_line_width = max(max_line_width, current_width)
                    current_line = [word]
                    current_width = word_width
                    total_lines += 1

            if current_line:
                max_line_width = max(max_line_width, current_width)
                total_lines += 1

        line_height = int(font_size * line_height_factor)
        total_height = total_lines * line_height
        
        return max_line_width, total_height

    def calculate(self, text, mask_width, mask_height, min_font_size, font, padding, line_height_factor):
        textbox_width = mask_width + (2 * padding)
        textbox_height = mask_height + (2 * padding)
        
        effective_width = textbox_width - 2 * padding
        effective_height = textbox_height - 2 * padding

        text_width, text_height = self.calculate_text_bounds(
            text, min_font_size, font, effective_width, line_height_factor
        )

        # Calculate ratios, considering growth in both directions (divide by 2)
        width_ratio = (text_width / mask_width) / 2
        height_ratio = (text_height / mask_height) / 2
        
        # Take the larger ratio, convert to percentage growth needed
        growth_ratio = max(width_ratio, height_ratio)
        
        # If ratio is less than 0.5 (1/2), no growth needed
        if growth_ratio <= 0.5:
            return (0,)
            
        # Convert to percentage (ratio of 1 = 100% growth needed)
        growth_percentage = min(100, int((growth_ratio - 0.5) * 200))
        
        return (growth_percentage,)

class BookToolsRandomTextOverlay:
    def __init__(self, device="cpu"):
        self.device = device
    _alignments = ["left", "center"]  # Removed "right"
    _positions = ["top", "bottom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "font_size": ("INT", {"default": 30, "min": 8, "max": 100, "step": 1}),
                "font": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}), 
                "color": ("STRING", {"default": "#000000"}),  
                "padding": ("INT", {"default": 20, "min": 0}),
                "margin": ("INT", {"default": 20, "min": 0}),
                "corner_radius": ("INT", {"default": 15, "min": 0}),
                "line_height_factor": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_random_text_overlay"
    CATEGORY = "image/text"

    def add_random_text_overlay(self, image, text, font_size, font, color, padding, margin, corner_radius, line_height_factor):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image_pil = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        
        # Convert image to RGBA for transparency support
        image_pil = image_pil.convert('RGBA')
        color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        
        # Create separate overlay for transparent background
        bg_overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_overlay)

        # Create separate overlay for opaque text
        text_overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_overlay)
        print(font)
        loaded_font = ImageFont.truetype(font, font_size)

        # Calculate dimensions and wrap text (existing code)
        max_width = image_pil.width - (margin * 2) - (padding * 2)
        line_height = int(font_size * line_height_factor)
        space_width = text_draw.textlength(" ", font=loaded_font)
        
        # Improved text wrapping that allows mid-word breaks if needed
        lines = []
        current_line = []
        current_width = 0
        
        for word in text.split():
            word_width = text_draw.textlength(word, font=loaded_font)
            
            # Add word if it fits, or split long words
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                # Handle words longer than max width
                if word_width > max_width:
                    chars = []
                    char_width = 0
                    for c in word:
                        cw = text_draw.textlength(c, font=loaded_font)
                        if char_width + cw > max_width:
                            lines.append("".join(chars))
                            chars = []
                            char_width = 0
                        chars.append(c)
                        char_width += cw
                    if chars:
                        lines.append("".join(chars))
                else:
                    current_line = [word]
                    current_width = word_width + space_width
                    
        if current_line:
            lines.append(" ".join(current_line))

        # Calculate total text block dimensions
        max_line_width = max(text_draw.textlength(line, font=loaded_font) for line in lines)
        total_height = len(lines) * line_height

        # Calculate background dimensions with padding only
        bg_width = int(max_line_width + padding * 2)  # Padding for text inside box
        bg_height = int(total_height + padding * 2)
        
        # Random position and alignment
        position = random.choice(self._positions)
        alignment = random.choice(self._alignments)

        # Calculate background position with margins
        if position == "top":
            bg_y = margin
            opacity = 0.9
        else:
            bg_y = image_pil.height - bg_height - margin
            opacity = 0.8

        if alignment == "left":
            bg_x = margin
        else:  # center
            bg_x = (image_pil.width - bg_width) // 2

        # Draw semi-transparent background
        alpha = int(255 * opacity)
        bg_color = (255, 255, 255, alpha)
        bg_draw.rounded_rectangle(
            [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
            radius=corner_radius,
            fill=bg_color
        )

        # Draw fully opaque text
        y = bg_y + padding
        for line in lines:
            line_width = text_draw.textlength(line, font=loaded_font)
            
            if alignment == "left":
                x = bg_x + padding
            else:  # center
                x = bg_x + (bg_width - line_width) // 2

            # Draw shadow first
            text_draw.text((x+1, y+1), line, fill=(255,255,255,150), font=loaded_font)  # Shadow
            # Draw main text
            text_draw.text((x, y), line, fill=(*color_rgb, 255), font=loaded_font)
            y += line_height

        # Composite in the right order: background first, then text
        result = Image.alpha_composite(image_pil, bg_overlay)
        result = Image.alpha_composite(result, text_overlay)
        
        # Convert back to RGB
        result = result.convert('RGB')
        
        image_tensor_out = torch.tensor(np.array(result).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        return (image_tensor_out,)

NODE_CLASS_MAPPINGS = {
    "BTPromptSelector": BookToolsPromptSelector,
    "BTPromptSchedule": BookToolsPromptSchedule,
    "Loop": BookToolsLoop,
    "LoopStart": BookToolsLoopStart,
    "LoopEnd": BookToolsLoopEnd,
    "EndQueue": BookToolsEndQueue,
    "ImageTextOverlay": BookToolsImageTextOverlay,
    "DownloadFont": BookToolsDownloadFont,
    "TextGrowth": BookToolsCalculateTextGrowth,
    "RandomTextOverlay": BookToolsRandomTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BTPromptSelector": "[Book Tools] Prompt Selector",
    "BTPromptSchedule": "[Book Tools] Prompt Batch Schedule",
    "Loop": "[Book Tools] Loop",
    "LoopStart": "[Book Tools] Loop Start",
    "LoopEnd": "[Book Tools] Loop End",
    "EndQueue": "[Book Tools] End Queue",
    "ImageTextOverlay": "[Book Tools] Image Text Overlay",
    "DownloadFont": "[Book Tools] Download Font",
    "TextGrowth": "[Book Tools] Calculate Text Growth",
    "RandomTextOverlay": "[Book Tools] Random Text Overlay",
}