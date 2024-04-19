from nodes import interrupt_processing
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class PromptSelector:
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
        - 'selected_indexes': a string listing keys by index, separated by commas, with optional ranges.
        """
        return {
            "required": {
                "dictionary": ("DICTIONARY", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "main"
    CATEGORY = "selector"
    OUTPUT_NODE = False

    def main(self, dictionary: dict, selected_indexes: str):
        if not isinstance(dictionary, dict):
            raise ValueError("Conditioning must be a dictionary.")

        selected_keys = selected_indexes.split(',')
        result_string = ', '.join(dictionary.get(key.strip(), '') for key in selected_keys)
        return (result_string,)

class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":"'first page description',\n'second page description'"}),
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
class Loop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("LOOP",)
    FUNCTION = "run"
    CATEGORY = "loopback"

    def run(self):
        return (self,)

class LoopStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "first_loop": (any,),
            "reset": ("BOOLEAN", {"default": False}),
            "loop": ("LOOP",)
            }
        }

    FUNCTION = "run"
    CATEGORY = "loopback"
    RETURN_TYPES = (any,)

    def run(self, first_loop, reset, loop):
        if (reset == True):
            return (first_loop,)
        if hasattr(loop, 'next'):
            return (loop.next,)
        return (first_loop,)

    @classmethod
    def IS_CHANGED(s, first_loop, reset, loop):
        if (reset == True):
            return (first_loop,)
        if hasattr(loop, 'next'):
            return id(loop.next)
        return float("NaN")

class LoopEnd:
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

class EndQueue:
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

class ImageTextOverlay:
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
                "max_font_size": ("INT", {"default": 30, "min": 1, "max": 256, "step": 1}),  
                "font": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}), 
                "alignment": (cls._alignments, {"default": "center"}),  
                "color": ("STRING", {"default": "#000000"}),  
                "start_x": ("INT", {"default": 0}),  
                "start_y": ("INT", {"default": 0}),
                "padding": ("INT", {"default": 50}),
                "line_height": ("INT", {"default": 20, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_text_overlay"
    CATEGORY = "image/text"

    def wrap_text_and_calculate_height(self, text, font, max_width, line_height):
        wrapped_lines = []
        # Split the input text by newline characters to respect manual line breaks
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            words = paragraph.split()
            current_line = words[0] if words else ''
            
            for word in words[1:]:
                # Test if adding a new word exceeds the max width
                test_line = current_line + ' ' + word if current_line else word
                test_line_bbox = font.getbbox(test_line)
                w = test_line_bbox[2] - test_line_bbox[0]  # Right - Left for width
                if w <= max_width:
                    current_line = test_line
                else:
                    # If the current line plus the new word exceeds max width, wrap it
                    wrapped_lines.append(current_line)
                    current_line = word
            
            # Don't forget to add the last line of the paragraph
            wrapped_lines.append(current_line)

        # Calculate the total height considering the custom line height
        total_height = len(wrapped_lines) * line_height

        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text, total_height


    def add_text_overlay(self, image, text, textbox_width, textbox_height, max_font_size, font, alignment, color, start_x, start_y, padding, line_height):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image_pil = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        color_rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        effective_textbox_width = textbox_width - 2 * padding  # Adjust for padding
        effective_textbox_height = textbox_height - 2 * padding

        font_size = max_font_size
        while font_size >= 1:
            loaded_font = ImageFont.truetype(font, font_size)
            wrapped_text, total_text_height = self.wrap_text_and_calculate_height(text, loaded_font, effective_textbox_width, line_height)

            if total_text_height <= effective_textbox_height:
                draw = ImageDraw.Draw(image_pil)
                lines = wrapped_text.split('\n')
                y = start_y + padding + (effective_textbox_height - total_text_height) // 2

                for line in lines:
                    line_bbox = loaded_font.getbbox(line)
                    line_width = line_bbox[2] - line_bbox[0]

                    if alignment == "left":
                        x = start_x + padding
                    elif alignment == "right":
                        x = start_x + effective_textbox_width - line_width + padding
                    elif alignment == "center":
                        x = start_x + padding + (effective_textbox_width - line_width) // 2

                    draw.text((x, y), line, fill=color_rgb, font=loaded_font)
                    y += line_height  # Use custom line height for spacing

                break  # Break the loop if text fits within the specified dimensions

            font_size -= 1  # Decrease font size and try again

        image_tensor_out = torch.tensor(np.array(image_pil).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        return (image_tensor_out,)
    
NODE_CLASS_MAPPINGS = {
    "PromptSelector": PromptSelector,
    "PromptSchedule": PromptSchedule,
    "Loop": Loop,
    "LoopStart": LoopStart,
    "LoopEnd": LoopEnd,
    "EndQueue": EndQueue,
    "ImageTextOverlay": ImageTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptSelector": "Prompt Selector",
    "PromptSchedule": "Prompt Batch Schedule",
    "LoopInt": "Loop",
    "LoopStartInt": "Loop Start",
    "LoopEndInt": "Loop End",
    "EndQueue": "End Queue",
    "ImageTextOverlay": "Image Text Overlay"
}